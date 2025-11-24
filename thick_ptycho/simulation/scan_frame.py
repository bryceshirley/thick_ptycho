from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass
class Point:
    """
    Defines a point.
    Either float or integer.
    Second dimension y is optional.
    """
    x: Union[float, int]
    y: Optional[Union[float, int]] = None

    def as_tuple(self) -> Tuple[float,...]:
        return (self.x,) if self.y is None else (self.x,self.y)

@dataclass
class Limits:
    """
    Defines Limits for a solver.
    Either float or integer.
    Second and third dimensions x and y are optional.
    """
    x: Union[Tuple[int, int],
             Tuple[float, float]]
    y: Optional[Union[Tuple[int, int], 
                      Tuple[float, float]]] = None
    z: Optional[Union[Tuple[int,int],
                       Tuple[float,float]]] = None
    units: str = "pixels"  # or "meters"

    def __post_init__(self):
        """Validate limits after initialization."""
        if self.units not in ("pixels", "meters","p" ,"m"):
            raise ValueError(f"units must be either 'pixels' or 'meters', got {self.units}")

        # Ensure tuples of correct type and length
        for coords in self.as_tuple():
            if not isinstance(coords, tuple) or len(coords) != 2:
                raise ValueError(f"Limits must be a tuple of length 2, got {coords}")

            if self.units in ["pixels","p"]:
                # Must be integers and non-negative
                if not all(isinstance(v, (int,np.integer)) for v in coords):
                    raise ValueError(f"Limits must be integers when units='pixels', got {coords}")
                if coords[0] < 0 or coords[1] < 0:
                    raise ValueError(f"Limits must be non-negative, got {coords}")
            elif self.units in ["meters","m"]:
                # Must be floats and positive or zero
                # Check numeric type
                if not all(isinstance(v, (int, float,np.integer, np.floating)) for v in coords):
                    raise ValueError(f"Limits must be numeric, got {coords}")
    
            # Increasing order
            if coords[0] >= coords[1]:
                raise ValueError(f"Limits must be increasing, got {coords}")

    def as_tuple(self) -> Tuple[Union[Tuple[int, int], Tuple[float, float]], ...]:
        if self.y is None and self.z is None:
            return (self.x,)
        elif self.z is None:
            return (self.x, self.y)
        elif self.y is None:
            return (self.x, self.z)
        else:
            return (self.x, self.y, self.z)

class ScanPath(Enum):
    """Enumeration of supported scan path patterns."""
    RASTER = "raster"
    SERPENTINE = "serpentine"
    SPIRAL = "spiral"

    @classmethod
    def list(cls):
        """Return a list of all scan path names."""
        return [p.value for p in cls]
    
@dataclass
class ScanFrame:
    """Represents one scan frame in a ptychographic scan."""
    probe_centre_continuous: Point
    probe_centre_discrete: Point
    reduced_limits_continuous: Optional[Limits] = None
    reduced_limits_discrete: Optional[Limits] = None

    def set_reduced_limits_continuous(self, limits: Limits):
        if limits.units not in ("meters", "m"):
            raise ValueError("reduced_limits_continuous must have units in 'meters'.")
        self.reduced_limits_continuous = limits
        return self
    
    def set_reduced_limits_discrete(self, limits: Limits):
        if limits.units not in ("pixels", "p"):
            raise ValueError("reduced_limits_discrete must have units in 'pixels'.")
        self.reduced_limits_discrete = limits
        return self
    
    def __post_init__(self):
        """Validate scan frame parameters after initialization."""
        if self.reduced_limits_continuous is not None:
            if self.reduced_limits_continuous.units != "meters":
                raise ValueError("reduced_limits_continuous must have units in 'meters'.")
        if self.reduced_limits_discrete is not None:
            if self.reduced_limits_discrete.units != "pixels":
                raise ValueError("reduced_limits_discrete must have units in 'pixels'.")

