"""
Factory for creating a PtychoObject2d or PtychoObject3d based on the simulation space.
"""

from .ptycho_object_2d import PtychoObject2D
from .ptycho_object_3d import PtychoObject3D


def create_ptycho_object(simulation_space):
    """
    Create a 2d or 3d ptychographic object depending on the simulation space.

    Parameters
    ----------
    simulation_space : object
        Simulation space instance that defines the spatial dimensionality.
        Must include an attribute `dimension` equal to 1 or 2.

    Returns
    -------
    PtychoObject2d or PtychoObject3d
        The appropriate object instance for the given simulation space.

    Raises
    ------
    ValueError
        If the simulation space dimension is not supported.
    """
    dim = getattr(simulation_space, "dimension", None)

    if dim == 2:
        return PtychoObject2D(simulation_space)
    elif dim == 3:
        return PtychoObject3D(simulation_space)

    raise ValueError(f"Unsupported simulation space dimension: {dim}")
