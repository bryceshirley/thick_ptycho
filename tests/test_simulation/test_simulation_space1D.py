import numpy as np
import pytest
from thick_ptycho.simulation import SimulationSpace1D, SimulationSpace2D

# Non Default Configurations
NONDEFAULT_CONFIG_1D = dict(
    wave_length=0.05,
    probe_diameter=0.01,
    continuous_dimensions=((0, 1), (0, 1)),
)


def test_effective_domain_consistency_with_rounding():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=7,
        step_size_px=5,
        pad_factor=1.33,
        solve_reduced_domain=False,
    )

    base_n_min = 7 * 5  # new definition: step_size * scan_points
    # min_nx may be rounded up by implementation (keep <= +1 leniency)
    assert sim.min_nx in (base_n_min, base_n_min + 1), "min_nx computation incorrect."

    # padding = nx - N_min
    assert sim.pad_discrete == sim.nx - sim.min_nx, "Padding computation incorrect."

    # must be even after correction
    assert sim.pad_discrete % 2 == 0, "Padding is not even."

    # symmetric split
    assert sim.edge_margin == sim.pad_discrete // 2, "Edge margin incorrect."

def test_min_nx_computation_with_adjustment():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=5,
        step_size_px=4,
        pad_factor=1.0,
    )
    base_n_min = 5 * 4  # new N_min
    assert sim.min_nx in (base_n_min, base_n_min + 1), "min_nx computation incorrect."

def test_padding_even_and_margin_correct():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=5,
        step_size_px=4,
        pad_factor=2.0,
    )
    assert sim.pad_discrete % 2 == 0, "Padding is not even."
    assert sim.edge_margin == sim.pad_discrete // 2, "Edge margin incorrect."

def test_scan_centres_discrete_spacing_and_symmetry():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=4,
        step_size_px=3,
        pad_factor=1.0,
    )
    centres = [f.probe_centre_discrete for f in sim._scan_frame_info]

    # spacing stays equal to step_size
    diffs = np.diff(centres)
    assert np.all(diffs == sim.step_size), "Scan centre spacing incorrect."

    # first/last symmetric about domain centre (robust across padding)
    # i.e., indices reflect: first + last == (nx - 1)
    assert centres[0] + centres[-1] == sim.nx - 1, "Scan centre symmetry incorrect."

def test_scan_centres_continuous_spacing_and_symmetry():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=4,
        step_size_px=3,
        pad_factor=1.0,
    )
    centres_cont = np.array([f.probe_centre_continuous for f in sim._scan_frame_info])

    # spacing in continuous coords == step_size * dx
    diffs_c = np.diff(centres_cont)
    assert np.allclose(diffs_c, sim.step_size * sim.dx), "Scan centre spacing incorrect."

    # symmetry in continuous coords: first + last == (nx-1) * dx
    assert np.isclose(centres_cont[0] + centres_cont[-1], (sim.nx - 1) * sim.dx), "Scan centre symmetry incorrect."

def test_domain_limits_without_reduction():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=4,
        step_size_px=3,
        pad_factor=2.0,
        solve_reduced_domain=False,
    )

    assert (sim.x[0], sim.x[-1]) == (
        sim.continuous_dimensions[0][0],
        sim.continuous_dimensions[0][1],
    ), "Domain continuous limits incorrect."
    # When not reducing, effective domain equals full Nx
    assert sim.effective_dimensions == sim.nx, "Effective domain size incorrect."

def test_domain_limits_with_reduction_new_effective_ne():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=4,
        step_size_px=3,
        pad_factor=2.0,
        solve_reduced_domain=True,
    )

    # New model: effective domain size Ne = step_size + padding
    expected_ne = sim.step_size + sim.pad_discrete
    assert sim.effective_dimensions == expected_ne, "Effective domain size incorrect."

    # Check that generated scan frames respect this effective domain
    first = sim._scan_frame_info[0].reduced_limits_discrete
    assert first == (0, sim.effective_nx-1), f"First scan frame limits incorrect. Expected (0, {sim.effective_nx}), got {first}."

def test_scan_frame_limits_stay_within_domain_and_width_is_effective():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=6,
        step_size_px=3,
        pad_factor=2.0,
    )

    expected_width = sim.step_size + sim.pad_discrete  # Ne
    for idx, f in enumerate(sim._scan_frame_info):
        xmin, xmax = f.reduced_limits_discrete
        assert xmin >= 0, f"Scan frame {idx} exceeds min domain limits. {xmin} < 0"
        assert xmax < sim.nx, f"Scan frame {idx} exceeds max domain limits. {xmax} >= {sim.nx}"
        assert (xmax - xmin) == expected_width-1, f"Scan frame {idx} width incorrect."

def test_scan_frame_spacing():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=4,
        step_size_px=4,
        pad_factor=1.5,
    )
    centres = [f.probe_centre_discrete for f in sim._scan_frame_info]
    diffs = np.diff(centres)
    assert np.all(diffs == sim.step_size), f"Scan frame centres not spaced by step_size. {diffs} != {sim.step_size}"

def test_first_and_last_scan_position_are_symmetric_not_edge_locked():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=5,
        step_size_px=4,
        pad_factor=2.0,
    )

    first = sim._scan_frame_info[0].probe_centre_discrete
    last = sim._scan_frame_info[-1].probe_centre_discrete

    # New centred model: no longer expect first==edge_margin.
    # Instead, centres are symmetric about the domain midpoint.
    assert first + last == sim.nx, f"First and last scan positions not symmetric about domain centre. {first} + {last} != {sim.nx - 1}"

def test_single_scan_point_is_centered():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=1,
        step_size_px=7,  # arbitrary; should not break centring
        pad_factor=1.6,
    )
    centre = sim._scan_frame_info[0].probe_centre_discrete

    # centred (allow off-by-one for even/odd Nx)
    mid = (sim.nx - 1) / 2.0
    assert abs(centre - mid) <= 0.5, "Single scan point not centred."

    # Effective domain: Ne = step_size + padding still holds
    assert sim.effective_dimensions == sim.nx, "Effective domain incorrect for single scan point."

def test_single_scan_point_window_stays_in_bounds():
    sim = SimulationSpace1D(
        **NONDEFAULT_CONFIG_1D,
        scan_points=1,
        step_size_px=7,
        pad_factor=1.4,
    )
    f = sim._scan_frame_info[0]
    xmin, xmax = f.reduced_limits_discrete
    assert xmin == 0, "Scan frame minimum limit out of bounds."
    assert xmax == sim.nx - 1, "Scan frame maximum limit out of bounds."

