"""
Thick Ptychography Simulation and Reconstruction Package.

Provides forward models, reconstruction algorithms, and simulation
utilities for thick-sample ptychographic imaging.
"""

__version__ = "0.1.0"

from .forward_model.multislice.ms_solver import MSForwardModelSolver
from .simulation.config import SimulationConfig, BoundaryType, ProbeType
from .simulation import ptycho_object, ptycho_probe, simulation_space
from .forward_model import BasePWESolver, PWEIterativeLUSolver, PWEFullPinTSolver
from .reconstruction import base_reconstructor, ms_reconstructor, pwe_reconstructor
from .utils import *

__all__ = [
    "SimulationConfig",
    "BoundaryType",
    "ProbeType",
    "ptycho_object",
    "ptycho_probe",
    "simulation_space",
    "MSForwardModelSolver",
    "PWEFullPinTSolver",
    "PWEIterativeLUSolver",
    "ms_reconstructor",
    "pwe_reconstructor",
    "utils",
]
