import numpy as np
import time
from typing import Optional, List, Literal, Any
from matplotlib import pyplot as plt
import os

from thick_ptycho.thick_ptycho.utils.visualisations import Visualisation
from thick_ptycho.utils.utils import setup_log


import numpy as np
import time
from typing import Optional, List, Literal, Any, Dict, Union

from thick_ptycho.thick_ptycho.simulation.ptycho_object import PtychoObject1D, PtychoObject2D
from thick_ptycho.thick_ptycho.simulation.simulation_space import SimulationSpace1D, SimulationSpace2D
from thick_ptycho.utils.utils import setup_log

# TODO: Update Visualisation of data and add support for tomographic projects
class BaseForwardModel:
    """
    Abstract base class for all forward model solvers (PWE, Multislice, etc.).
    Handles:
      - Logging
      - Probe generation
      - Common solve() interface across all solvers
      - Multiple probe and angle looping
      - Simulated data creation (exit waves + noisy farfield intensities)
    """

    def __init__(
        self,
        simulation_space: Union[SimulationSpace1D, SimulationSpace2D],
        ptycho_object: Union[PtychoObject1D, PtychoObject2D],
        ptycho_probes: np.ndarray,
        results_dir: str = "",
        use_logging: bool = False,
        verbose: bool = True,
        log=None,
    ):
        self.simulation_space = simulation_space
        self.ptycho_object = ptycho_object
        self.verbose = verbose
        self.results_dir = results_dir

        if results_dir is None:
            self.results_dir = simulation_space.results_dir

        # Logger
        self._log = log or setup_log(results_dir, "solver_log.txt", use_logging, verbose)

        # Determine slice dimensions
        self.thin_sample = simulation_space.thin_sample
        self.slice_dimensions = simulation_space.slice_dimensions
        self.nz = simulation_space.nz

        # Probe setup
        self.probes = ptycho_probes
        self.num_probes = simulation_space.num_probes
        self.num_projections = simulation_space.num_projections
        assert self.num_projections <= 2, "Only supports up to 2 projections."
        self.probe_angles_list = simulation_space.probe_angles_list
        self.num_angles = len(self.probe_angles_list)

        # Solver type (for logging purposes)
        self.solver_type = "BaseForwardModel"

        # Visualization utility
        self.visualisation = Visualisation(self.simulation_space, results_dir=self.results_dir)

    # ------------------------------------------------------------------
    # Common solving interface
    # ------------------------------------------------------------------

    def solve(self, n: Optional[np.ndarray] = None, mode: str = "forward",
              rhs_block: Optional[np.ndarray] = None,
              initial_condition: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Main multi-angle, multi-probe solving loop.
        Subclasses must define `_solve_single_probe(angle_idx, probe_idx, n, **kwargs)`.
        """
        # Initialize solution grid with initial condition
        u = self._create_solution_grid()

        # Loop over angles and probes
        for proj_idx in range(self.num_projections):
            for angle_idx, angle in enumerate(self.probe_angles_list):
                for scan_idx in range(self.num_probes):
                    start = time.time()

                    # Solve for single probe
                    u[proj_idx, angle_idx, scan_idx, ...] = self._solve_single_probe(
                        proj_idx=proj_idx, angle_idx=angle_idx, scan_idx=scan_idx,
                        n=n, mode=mode, rhs_block=rhs_block,
                        initial_condition=initial_condition,
                    )

                    # Log time taken for each probe if verbose is True
                    if self.verbose:
                        self._log(
                            f"[{self.solver_type}] solved probe {scan_idx+1}/{self.num_probes} "
                            f"at angle {angle} in {time.time() - start:.2f}s"
                        )
        return u

    def _solve_single_probe(self, angle_idx: int, probe_idx: int,
                             n=None, mode: str = "forward", initial: np.ndarray = None) -> np.ndarray:
        """Override in subclasses."""
        raise NotImplementedError

    def _create_solution_grid(self) -> np.ndarray:
        """Create an empty solution tensor."""
        return np.zeros(
            (self.num_projections, self.num_angles, self.num_probes, *self.slice_dimensions, self.nz),
            dtype=complex,
        )
    
    def rotate_n(self, n: np.ndarray) -> np.ndarray:
        """Rotate the refractive index array by 90 degrees."""
        assert self.simulation_space.dimension == 1, "rotate_n only supports 1D images."
        return np.rot90(n, k=1)

    # ------------------------------------------------------------------
    # Synthetic data generation utilities
    # ------------------------------------------------------------------

    def get_exit_waves(self, u: Optional[np.ndarray] = None, save_plots: bool = False) -> np.ndarray:
        """
        Simulate exit waves for all probes and angles.

        Parameters
        ----------
        u : np.ndarray, optional
            Precomputed field inside the sample. If None, will be computed via `solve()`.
        save_plots : bool, default False
            Whether to save exit wave plots.

        Returns
        -------
        exit_waves : np.ndarray
            Exit wave field at detector plane (z = nz - 1)
            Shape: (num_angles, num_probes, *slice_dimensions)
        """
        # Slice final z-plane for each angle & probe
        if self.simulation_space.dimension == 1:
            exit_waves = u[..., -1]
        elif self.simulation_space.dimension == 2:
            exit_waves = u[..., -1]
        else:
            raise ValueError("Unsupported sample dimension for exit wave simulation")
        
        if save_plots:
            self.save_exit_wave_plots(exit_waves)
        return exit_waves

    def get_farfield_intensities(
        self,
        u: Optional[np.ndarray] = None,
        exit_waves: Optional[np.ndarray] = None,
        poisson_noise: Optional[bool] = True,
        save_plots: bool = False,
    ) -> np.ndarray:
        """
        Simulate noisy far-field diffraction intensities.

        Parameters
        ----------
        u : np.ndarray, optional
            Precomputed field inside the sample. If None, will be computed via `solve()`.
        exit_waves : np.ndarray, optional
            Precomputed exit waves. If None, will be computed via `simulate_exit_waves()`.
        noise_model : str, default "poisson"
            Type of noise: "poisson", "gaussian", "mixed", or None.
        snr_db : float, optional
            Signal-to-noise ratio (for Gaussian noise only).
        normalize : bool, default True
            Normalize intensity patterns to unit mean.

        Returns
        -------
        intensities: np.ndarray
        """
        exit_waves = self.get_exit_waves(u=u, save_plots=save_plots) if exit_waves is None else exit_waves

        # Compute FFTs for all probes and angles
        if self.simulation_space.dimension == 1:
            fft_waves = np.fft.fftshift(np.fft.fft(exit_waves, axis=-1))
        else:
            fft_waves = np.fft.fftshift(np.fft.fft2(exit_waves, axes=(-2, -1)))

        intensities = np.abs(fft_waves) ** 2

        # Add noise
        if poisson_noise:
            intensities = np.random.poisson(intensities)
        
        if save_plots:
            self.save_intensity_plot(intensities)
        return intensities