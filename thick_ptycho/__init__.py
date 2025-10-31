"""
Thick Ptychography Simulation and Reconstruction Package.

Provides forward models, reconstruction algorithms, and simulation
utilities for thick-sample ptychographic imaging.
"""

__version__ = "0.1.0"

from .simulation.config import SimulationConfig, BoundaryType, ProbeType
from .simulation import ptycho_object, ptycho_probe, simulation_space
from .forward_model import base_solver, ms_solver, pwe_solver_full_pint, pwe_solver_iterative
from .reconstruction import base_reconstructor, ms_reconstructor, pwe_reconstructor
from . import utils

__all__ = [
    "SimulationConfig",
    "BoundaryType",
    "ProbeType",
    "ptycho_object",
    "ptycho_probe",
    "simulation_space",
    "ms_solver",
    "pwe_solver_full_pint",
    "pwe_solver_iterative",
    "ms_reconstructor",
    "pwe_reconstructor",
    "utils",
]
