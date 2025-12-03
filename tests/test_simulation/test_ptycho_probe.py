import pytest

from thick_ptycho.simulation import create_simulation_space, SimulationConfig
from thick_ptycho.simulation.ptycho_probe import PtychoProbes

NONDEFAULT_CONFIG = dict(
    wave_length=0.05,
    probe_diameter=0.01)


def sim_space(solve_reduced_domain, 
              spatial_limits,
              scan_points, step_size, pad):
    sim_config = SimulationConfig(
            **NONDEFAULT_CONFIG,
            spatial_limits=spatial_limits,
            solve_reduced_domain=solve_reduced_domain,
            scan_points=scan_points,
            step_size_px=step_size,
            pad_factor=pad,
            )
    return create_simulation_space(sim_config)


# def test_empty_probe_stack():
#     probes
