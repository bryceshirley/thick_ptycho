# Discretization of the Simulation Domain (1D Example)

This section explains how the discrete simulation grid is constructed
from the scanning parameters. The same principles generalize to 2D and 3D.

## Overview

A 1D sample domain is discretized into Nx points with spacing ``dx``.
A probe scans the sample at scan_points (or in diagram ``n``) positions,
separated by step_size_px (or in diagram ``s``) pixels.
To ensure the full probe footprint fits inside the simulated region at
each position, the domain must be padded on both sides.

## Diagram

    x = 0                                                                  x = Nx * dx
     ____________________________________________________________________________
     |<-- pL -->|<--  s  -->|<--  s  -->|   ...  |<-- s -->|<-- s -->|<-- pR -->|
                      c₁          c₂                  cₙ₋₁       cₙ
                      |<--  s  -->|
                 |<-- pL -->|<--  s  -->|<--  pR  -->|
                 |<------------  Ne  --------------->|
                |<------------------------ min_nx ------------------>|
     |<------------------------------        Nx       -------------------------->|
Where:
    cᵢ  : Scan point centers (pixel indices)
    s   : Step size between consecutive scan points (pixels)
    n   : Number of scan positions
    min_nx : Minimum required simulation width to contain all scan positions
    Nx  : Total number of grid points in the simulation domain (including padding)
    pL, pR : Padding widths on the left and right sides (pL = pR)
    Ne  : Effective padding region width (pL + pR)

### Key Quantities

step_size_px : int
    Step size between scan positions, measured in pixels.

scan_points : int
    Number of probe positions along the scan line.

min_nx : int
    Minimum required simulation width to contain all scan positions.
    Computed as::
        min_nx = scan_points * step_size_px

pad_factor : float, >= 1.0
    Controls how much total padding to add around the scanned region.
    A value of ``1.0`` means no padding; larger values expand the domain.

Nx : int
    Total number of grid points in the simulation domain, including padding.
    Computed as::

        Nx = int(pad_factor * min_nx) = int(pad_factor * scan_points * step_size_px)

Ne : int
    Effective . Used when solving only a reduced "effective" domain instead
    of the full padded space::
        padding = Nx - min_nx = Nx - (scan_points * step_size_px)
        Ne = step_size_px + padding

        alternatively, future refactoring:
        padding = int(pad_factor*step_size) = max_overlap of scan frames
        => Nx = scan_points * step_size_px + padding 

        ~~----|----~~
                 ~~----|----~~      

        padding is also the maximum overlap.

dx : float
    Spatial step in meters (physical pixel size).

### Notes

- A larger `pad_factor` reduces boundary artifacts and improves numerical stability,
  but increases memory and computational cost.
- Increasing `step_size_px` increases the resolution of the scan.
- When performing iterative reconstruction methods (e.g., PWE or MS),
  the effective domain width ``Ne`` may be used to accelerate computation considering
  a smaller region of interest.
- The discretization parameters should be chosen based on the physical
  dimensions of the sample and the probe characteristics to ensure accurate simulation results.