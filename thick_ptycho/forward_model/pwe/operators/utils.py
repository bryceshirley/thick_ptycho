from enum import Enum


class BoundaryType(Enum):
    """
    Supported boundary condition types for the simulation domain.
    Not required for Multislice Solver.
    """
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    IMPEDANCE = "impedance"
    IMPEDANCE2 = "impedance2"

    @classmethod
    def list(cls):
        """Return a list of all boundary condition names."""
        return [bc.value for bc in cls]