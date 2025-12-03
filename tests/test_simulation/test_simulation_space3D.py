import numpy as np
import pytest

from thick_ptycho.simulation import SimulationSpace3D
from thick_ptycho.simulation.scan_frame import Limits, ScanPath

NONDEFAULT_CONFIG_3D = dict(
    wave_length=0.05,
    probe_diameter=0.01,
    spatial_limits=Limits(x=(0, 1), y=(0, 1),z=(0, 1), units="meters"),
)

# --- Utility functions ---------------------------------------------------------

def _extract_centres(sim):
    """Return array of (cx, cy) discrete centres."""
    return np.array([f.probe_centre_discrete.as_tuple() for f in sim._scan_frame_info])


# --- Tests --------------------------------------------------------------------

def test_rotation():
    """
    No 90-degree tomographic projection in 3D.
    3D simulation should leave nz independent of nx.
    And num_projections should be 1.
    """
    sim = SimulationSpace3D(
        **NONDEFAULT_CONFIG_3D,
        tomographic_projection_90_degree=True,
    )
    assert sim.nz != sim.nx
    assert sim.num_projections == 1

@pytest.mark.parametrize(
    "scan_points, step_size, pad",
    [
        (4, 3, 1.2),
        (5, 4, 1.0),
        (5, 4, 2.0),
    ]
)
def test_domain_square_and_effective(scan_points, step_size, pad):
    sim = SimulationSpace3D(
        **NONDEFAULT_CONFIG_3D,
        solve_reduced_domain=True,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
    )

    # Full domain square
    assert sim.nx == sim.ny

    # Effective domain square
    assert sim.effective_nx == sim.effective_ny

    # Reduced dimension matches 1D rule
    expected_effective = sim.step_size + sim.pad_discrete - 1
    assert sim.effective_nx == expected_effective


@pytest.mark.parametrize(
    "scan_points, step_size, pad",
    [
        (4, 3, 1.2),
        (4, 3, 1.6),
    ]
)
def test_centres_form_uniform_grid_and_are_symmetric(scan_points, step_size, pad):
    sim = SimulationSpace3D(
        **NONDEFAULT_CONFIG_3D,
        solve_reduced_domain=True,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
    )

    centres = _extract_centres(sim)
    xs = np.unique(centres[:, 0])
    ys = np.unique(centres[:, 1])

    # Uniform grid spacing
    assert np.all(np.diff(xs) == sim.step_size)
    assert np.all(np.diff(ys) == sim.step_size)

    # Symmetric about domain mid
    mid = (sim.nx - 1) / 2
    assert abs(xs[0] - (mid - np.floor((len(xs) - 1) * sim.step_size / 2))) <= 0.5
    assert abs(ys[0] - (mid - np.floor((len(ys) - 1) * sim.step_size / 2))) <= 0.5


@pytest.mark.parametrize("scan_points, step_size, pad", [
    (4, 3, 1.5),
    (5, 4, 2.0),
])
def test_serpentine_path_order(scan_points, step_size, pad):
    sim = SimulationSpace3D(
        **NONDEFAULT_CONFIG_3D,
        solve_reduced_domain=True,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
        scan_path=ScanPath.SERPENTINE,
    )

    centres = _extract_centres(sim)

    nx = ny = scan_points  # your implementation only supports square grids
    centres_grid = centres.reshape(nx, ny, 2)  # <── column-major layout

    # Loop over columns (x fixed, y varies)
    for col_idx in range(nx):
        col = centres_grid[col_idx, :, 1]  # take y-coordinates in this column

        if col_idx % 2 == 0:
            assert np.all(np.diff(col) > 0)   # bottom → top
        else:
            assert np.all(np.diff(col) < 0)   # top → bottom

@pytest.mark.parametrize("scan_points, step_size, pad", [
    (4, 3, 1.5),
    (5, 4, 2.0),
])
def test_raster_path_order(scan_points, step_size, pad):
    sim = SimulationSpace3D(
        **NONDEFAULT_CONFIG_3D,
        solve_reduced_domain=True,
        scan_points=scan_points,
        step_size_px=step_size,
        pad_factor=pad,
        scan_path=ScanPath.RASTER,
    )

    centres = _extract_centres(sim)

    nx = ny = scan_points  # currently only square supported
    # For raster scan, interpretation is *row-major*:
    # y increases outer, x increases inner
    centres_grid = centres.reshape(ny, nx, 2)

    for row_idx in range(ny):
        row_x = centres_grid[row_idx, :, 0]
        # Must always increase left → right
        assert np.all(np.diff(row_x) > 0), f"Row {row_idx} is not increasing in raster mode"



# @pytest.mark.parametrize("scan_points, step_size, pad", [
#     (4, 3, 1.5),
# ])
# def test_spiral_path_has_same_centres_but_reordered(scan_points, step_size, pad):
#     serp = SimulationSpace3D(
#         **NONDEFAULT_CONFIG_3D,
#         solve_reduced_domain=True,
#         scan_points=scan_points,
#         step_size_px=step_size,
#         pad_factor=pad,
#         scan_path=ScanPath.SERPENTINE,
#     )
#     spir = SimulationSpace3D(
#         **NONDEFAULT_CONFIG_3D,
#         solve_reduced_domain=True,
#         scan_points=scan_points,
#         step_size_px=step_size,
#         pad_factor=pad,
#         scan_path=ScanPath.SPIRAL,
#     )

#     serp_centres = _extract_centres(serp)
#     spir_centres = _extract_centres(spir)

#     # --- Invariant 1: both scans visit the same centre coordinates ---
#     assert set(map(tuple, serp_centres)) == set(map(tuple, spir_centres))

#     # --- Invariant 2: Spiral peels inward (shrinking bounding box) ---
#     xs, ys = spir_centres[:, 0], spir_centres[:, 1]

#     # Track bounding box evolution along the spiral sequence
#     cum_min_x = np.minimum.accumulate(xs)
#     cum_max_x = np.maximum.accumulate(xs)
#     cum_min_y = np.minimum.accumulate(ys)
#     cum_max_y = np.maximum.accumulate(ys)

#     # Bounding box must only get tighter, never expand again
#     assert np.all(np.diff(cum_min_x) >= 0)
#     assert np.all(np.diff(cum_min_y) >= 0)
#     assert np.all(np.diff(cum_max_x) <= 0)
#     assert np.all(np.diff(cum_max_y) <= 0)



def test_single_scan_point_centering_and_full_window():
    sim = SimulationSpace3D(
        **NONDEFAULT_CONFIG_3D,
        solve_reduced_domain=True,
        scan_points=1,
        step_size_px=6,
        pad_factor=1.4,
    )
    c = sim._scan_frame_info[0].probe_centre_discrete
    mid = (sim.nx - 1) / 2
    assert abs(c.x - mid) <= 0.5
    assert abs(c.y - mid) <= 0.5

    xmin, xmax = sim._scan_frame_info[0].reduced_limits_discrete.x
    ymin, ymax = sim._scan_frame_info[0].reduced_limits_discrete.y
    assert (xmin, xmax) == (0, sim.nx - 1)
    assert (ymin, ymax) == (0, sim.ny - 1)


def test_domain_limits_without_reduction():
    sim = SimulationSpace3D(
        **NONDEFAULT_CONFIG_3D,
        scan_points=4,
        step_size_px=3,
        pad_factor=2.0,
        solve_reduced_domain=False,
    )
    # Whole domain is used
    assert sim.effective_nx == sim.nx
    assert sim.effective_ny == sim.ny


def test_domain_limits_with_reduction():
    sim = SimulationSpace3D(
        **NONDEFAULT_CONFIG_3D,
        scan_points=4,
        step_size_px=3,
        pad_factor=2.0,
        solve_reduced_domain=True,
    )
    expected_ne = sim.step_size + sim.pad_discrete - 1
    assert sim.effective_nx == expected_ne
    assert sim.effective_ny == expected_ne 

    xmin, xmax = sim._scan_frame_info[-1].reduced_limits_discrete.x
    assert (xmin, xmax) == (0, expected_ne)
    ymin, ymax = sim._scan_frame_info[-1].reduced_limits_discrete.y
    assert (ymin, ymax) == (0, expected_ne)

    assert sim.effective_shape == (sim.effective_nx, sim.effective_ny ,sim.nz)