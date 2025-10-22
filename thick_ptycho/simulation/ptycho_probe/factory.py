"""
Factory for creating a PtychoObject1D or PtychoObject2D based on the simulation space.
"""

from .ptycho_probe import PtychoProbes

def create_ptycho_probes(simulation_space):
    """
    Create a ptychographic probe set of shape (num_angles, num_probes, detector_dim1, [detector_dim2])
    based on the simulation space.
    """
    ptycho_probes = PtychoProbes(simulation_space)
    return ptycho_probes.build_probes()
