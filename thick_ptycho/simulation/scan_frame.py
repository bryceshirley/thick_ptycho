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
        for coords_name, coords in {"x": self.x, "y": self.y, "z": self.z}.items():
            if coords is None:
                continue

            if not isinstance(coords, tuple) or len(coords) != 2:
                raise ValueError(f"{coords_name} limits must be a tuple of length 2, got {coords}")

            if self.units == "pixels" or self.units=="p":
                # Must be integers and non-negative
                if not all(isinstance(v, (int,np.integer)) for v in coords):
                    raise ValueError(f"{coords_name} limits must be integers when units='pixels', got {coords}")
                if coords[0] < 0 or coords[1] < 0:
                    raise ValueError(f"{coords_name} limits must be non-negative, got {coords}")
            elif self.units == "meters" or self.units=="m":
                # Must be floats and positive or zero
                # Check numeric type
                if not all(isinstance(v, (int, float,np.integer, np.floating)) for v in coords):
                    raise ValueError(f"{coords_name} limits must be numeric, got {coords}")
            
            # Increasing order
            if coords[0] >= coords[1]:
                raise ValueError(f"{coords_name} limits must be increasing, got {coords}")

    def as_tuple(self) -> Tuple[Union[Tuple[int, int], Tuple[float, float]], ...]:
        if self.y is None and self.z is None:
            return (self.x,)
        elif self.z is None:
            return (self.x, self.y)
        elif self.y is None:
            return (self.x, self.z)
        else:
            return (self.x, self.y, self.z)


    # def __post_init__(self):
    #     """Validate limits after initialization."""

    #     # Convert integer limits to float if units are in meters
    #     try:
    #         if self.units == "meters":
    #             self.x = tuple(float(v) for v in self.x)
    #             if self.y is not None:
    #                 self.y = tuple(float(v) for v in self.y)
    #             if self.z is not None:
    #                 self.z = tuple(float(v) for v in self.z)
    #     except TypeError:
    #         raise ValueError("Limits must be tuples of two numeric values.")

    #     # Validate limits
    #     for coords in [self.x, self.y, self.z]:
    #         if coords is None:
    #             continue
    #         if len(coords) != 2:
    #             raise ValueError(f"limits must be tuples of length 2, got {coords}")
    #         if coords[0] >= coords[1]:
    #             raise ValueError(f"limits must be increasing, got {coords}")
    #         if self.units == "pixels":
    #             if not isinstance(coords[0], int) and not isinstance(coords[1], int):
    #                 ValueError(f"limits must be integers when units='pixels', got {self.x}")
    #             if coords[0] < 0 or coords[1] < 0:
    #                 raise ValueError(f"limits must be non-negative when units='pixels', got {self.x}")
    #         elif self.units == "meters":
    #             continue
    #         else:
    #             raise ValueError(f"units must be either 'pixels' or 'meters', got {self.units}")
        

    # def as_tuple(self) -> Tuple[Union[Tuple[int, int], Tuple[float, float]], ...]:
    #     if self.y is None and self.z is None:
    #         return (self.x,)
    #     elif self.z is None:
    #         return (self.x, self.y)
    #     elif self.y is None:
    #         return (self.x, self.z)
    #     else:
    #         return (self.x, self.y, self.z)

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

