import numpy as np
import time
from typing import Optional, Union

from thick_ptycho.simulation.ptycho_object import PtychoObject1D, PtychoObject2D
from thick_ptycho.simulation.simulation_space import SimulationSpace1D, SimulationSpace2D
from thick_ptycho.utils.io import setup_log

# TODO: Update Visualisation of data and add support for tomographic projects
class BaseForwardModelSolver:
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
        self.effective_dimensions = simulation_space.effective_dimensions
        self.nz = simulation_space.nz

        # Probe setup
        self.probes = ptycho_probes
        self.num_probes = simulation_space.num_probes
        self.num_projections = simulation_space.num_projections
        assert self.num_projections <= 2, "Only supports up to 2 projections."
        self.probe_angles = simulation_space.probe_angles
        self.num_angles = len(self.probe_angles)

        # Solver type (for logging purposes)
        self.solver_type = "BaseForwardModel"

    # ------------------------------------------------------------------
    # Common solving interface
    # ------------------------------------------------------------------

    def solve(self, n: Optional[np.ndarray] = None, mode: str = "forward",
              rhs_block: Optional[np.ndarray] = None,
              probes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Main multi-angle, multi-probe solving loop.
        Subclasses must define `_solve_single_probe(angle_idx, probe_idx, n, **kwargs)`.

        Parameters
        ----------
        n : ndarray, optional
            Refractive index distribution on the (x[,y], z) grid.
        mode : {'forward', 'adjoint','reverse'}
            Propagation mode.
        rhs_block : ndarray, optional
            Optional RHS vector for reusing precomputed blocks.
        initial_condition : ndarray, optional
            Initial probe condition to use instead of default.
        
        Returns
        -------
        u : ndarray
            Complex propagated field, shape
            (num_projections, num_angles, num_probes, nx[,ny], nz).
        """
        assert mode in {"forward", "adjoint","reverse"}, f"Invalid mode: {mode!r}"  
        # Initialize solution grid with initial condition
        u = self._create_solution_grid()

        if probes is None:
            probes = self.probes

        # Loop over angles and probes
        for proj_idx in range(self.num_projections):
            temp_mode = mode + "_rotated" if proj_idx == 1 else mode
    
            for angle_idx, angle in enumerate(self.probe_angles):
                for scan_idx in range(self.num_probes):
                    if self.verbose:
                        start = time.time()

                    # Solve for single probe
                    u[proj_idx, angle_idx, scan_idx, ...] = self._solve_single_probe(
                        scan_idx=scan_idx,
                        n=n, mode=temp_mode, 
                        rhs_block=rhs_block,
                        probe=probes[angle_idx, scan_idx, :],
                    )

                    # Log time taken for each probe if verbose is True
                    if self.verbose:
                        self._log(
                            f"[{self.solver_type}] solved probe {scan_idx+1}/{self.num_probes} "
                            f"at angle {angle} in {time.time() - start:.2f}s"
                        )
        return u

    
    def solve_batch(self, n=None, mode="forward", rhs_block=None, probes=None):
        """
        Batch solving interface (for future use).
        """
        assert mode in {"forward", "adjoint", "reverse"}

        u = self._create_solution_grid()
        probes = probes if probes is not None else self.probes
        B = self.num_angles * self.num_probes  # total probes in batch

        for proj_idx in range(self.num_projections):
            temp_mode = mode + "_rotated" if proj_idx == 1 else mode

            if self.verbose:
                start = time.time()

            # Flatten probe batch: (num_angles, num_probes, block_size) → (B, block_size)
            probe_batch = probes.reshape(B, self.block_size)

            # Batch solve
            result = self._solve_probes_batch(
                probes=probe_batch,
                n=n,
                mode=temp_mode,
                rhs_block=rhs_block,
            )  # result shape: (B, block_size, nz)

            # Unflatten: (B, block_size, nz) → (num_angles, num_probes, block_size, nz)
            u[proj_idx, ...] = result.reshape(self.num_angles, self.num_probes, self.block_size, self.nz)

            if self.verbose:
                self._log(
                    f"[{self.solver_type}] solved {B} probes (angles × scans) "
                    f"in {time.time() - start:.2f}s"
                )

        return u

    

    def _solve_single_probe(self, angle_idx: int, probe_idx: int,
                             n=None, mode: str = "forward", initial: np.ndarray = None) -> np.ndarray:
        """Override in subclasses."""
        raise NotImplementedError

    def _create_solution_grid(self) -> np.ndarray:
        """Create an empty solution tensor."""
        return np.zeros(
            (self.num_projections, self.num_angles, self.num_probes, *self.effective_dimensions, self.nz),
            dtype=complex,
        )
    
    def rotate_n(self, n: np.ndarray) -> np.ndarray:
        """Rotate the refractive index array by 90 degrees."""
        assert self.simulation_space.dimension == 1, "rotate_n only supports 1D images."
        return np.rot90(n, k=1)

    # ------------------------------------------------------------------
    # Synthetic data generation utilities
    # ------------------------------------------------------------------

    def get_exit_waves(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simulate exit waves for all probes and angles.

        Parameters
        ----------
        u : np.ndarray, optional
            Precomputed field inside the sample. If None, will be computed via `solve()`.

        Returns
        -------
        exit_waves : np.ndarray
            Exit wave field at detector plane (z = nz - 1)
            Shape: (num_angles, num_probes, *effective_dimensions)
        """
        return u[..., -1].reshape((self.simulation_space.total_scans, 
                       self.simulation_space.block_size))


    def get_farfield_intensities(
        self,
        u: Optional[np.ndarray] = None,
        exit_waves: Optional[np.ndarray] = None,
        poisson_noise: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Simulate noisy far-field diffraction intensities.

        Parameters
        ----------
        u : np.ndarray, optional
            Precomputed field inside the sample. If None, will be computed via `solve()`.
        exit_waves : np.ndarray, optional
            Precomputed exit waves. If None, will be computed via `get_exit_waves()`.
        poisson_noise : bool, default True
            If True, add Poisson noise to the intensities.
        Returns
        -------
        intensities: np.ndarray
        """
        exit_waves = self.get_exit_waves(u=u) if exit_waves is None else exit_waves

        # Compute FFTs for all probes and angles
        if self.simulation_space.dimension == 1:
            fft_waves = np.fft.fft(exit_waves)
        else:
            fft_waves = np.fft.fft2(exit_waves, axes=(-2, -1))

        intensities = np.abs(fft_waves) ** 2

        # Add noise
        if poisson_noise:
            intensities = np.random.poisson(intensities)
        return intensities.reshape((self.simulation_space.total_scans, 
                       self.simulation_space.block_size))