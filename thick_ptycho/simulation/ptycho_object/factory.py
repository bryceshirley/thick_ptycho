"""
Factory for creating a PtychoObject1D or PtychoObject2D based on the simulation space.
"""

from .ptycho_object_1d import PtychoObject1D
from .ptycho_object_2d import PtychoObject2D


def create_ptycho_object(simulation_space):
    """
    Create a 1D or 2D ptychographic object depending on the simulation space.

    Parameters
    ----------
    simulation_space : object
        Simulation space instance that defines the spatial dimensionality.
        Must include an attribute `dimension` equal to 1 or 2.

    Returns
    -------
    PtychoObject1D or PtychoObject2D
        The appropriate object instance for the given simulation space.

    Raises
    ------
    ValueError
        If the simulation space dimension is not supported.
    """
    dim = getattr(simulation_space, "dimension", None)

    if dim == 1:
        return PtychoObject1D(simulation_space)
    elif dim == 2:
        return PtychoObject2D(simulation_space)

    raise ValueError(f"Unsupported simulation space dimension: {dim}")
