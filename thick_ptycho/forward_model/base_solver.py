import numpy as np
import time
from typing import Optional, List, Literal, Any

from thick_ptycho.thick_ptycho.simulation.ptycho_object import SampleSpace
from thick_ptycho.utils.utils import setup_log
from thick_ptycho.thick_ptycho.simulation.ptycho_probe.ptycho_probe import Probes

import numpy as np
import time
from typing import Optional, List, Literal, Any, Dict, Union

from thick_ptycho.thick_ptycho.simulation.ptycho_object import PtychoObject1D, PtychoObject2D
from thick_ptycho.thick_ptycho.simulation.simulation_space import SimulationSpace1D, SimulationSpace2D
from thick_ptycho.utils.utils import setup_log
from thick_ptycho.thick_ptycho.simulation.ptycho_probe.ptycho_probe import PtychoProbes

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

        if results_dir is None:
            results_dir = simulation_space.results_dir

        # Logger
        self._log = log or setup_log(results_dir, "solver_log.txt", use_logging, verbose)

        # Determine slice dimensions
        self.thin_sample = simulation_space.thin_sample
        self.slice_dimensions = simulation_space.slice_dimensions
        self.nz = simulation_space.nz

        # Probe setup
        self.probes = ptycho_probes
        self.num_probes = simulation_space.num_probes
        self.probe_angles_list = simulation_space.probe_angles_list
        self.num_angles = len(self.probe_angles_list)

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
        for a_idx, angle in enumerate(self.probe_angles_list):
            for p_idx in range(self.num_probes):
                start = time.time()

                # Solve for single probe
                u[a_idx, p_idx, ...] = self._solve_single_probe(a_idx, p_idx,
                    n=n, mode=mode, rhs_block=rhs_block,
                    initial_condition=initial_condition
                )

                # Log time taken for each probe if verbose is True
                if self.verbose:
                    self._log(
                        f"[{self.solver_type}] solved probe {p_idx+1}/{self.num_probes} "
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
            (self.num_angles, self.num_probes, *self.slice_dimensions, self.nz),
            dtype=complex,
        )

    # ------------------------------------------------------------------
    # Synthetic data generation utilities
    # ------------------------------------------------------------------

    def simulate_exit_waves(self, n: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Simulate exit waves for all probes and angles.

        Parameters
        ----------
        n : np.ndarray, optional
            Refractive index field used for forward propagation.

        Returns
        -------
        exit_waves : np.ndarray
            Exit wave field at detector plane (z = nz - 1)
            Shape: (num_angles, num_probes, *slice_dimensions)
        """
        u = self.solve(n=n, **kwargs)
        # Slice final z-plane for each angle & probe
        if self.simulation_space.dimension == 1:
            return u[..., -1]
        elif self.simulation_space.dimension == 2:
            return u[..., -1]
        else:
            raise ValueError("Unsupported sample dimension for exit wave simulation")

    def simulate_farfield_intensities(
        self,
        n: Optional[np.ndarray] = None,
        exit_waves: Optional[np.ndarray] = None,
        poisson_noise: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Simulate noisy far-field diffraction intensities.

        Parameters
        ----------
        n : np.ndarray, optional
            Refractive index field.
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
        exit_waves = exit_waves or self.simulate_exit_waves(n=n)

        # Compute FFTs for all probes and angles
        if self.simulation_space.dimension == 1:
            fft_waves = np.fft.fftshift(np.fft.fft(exit_waves, axis=-1))
        else:
            fft_waves = np.fft.fftshift(np.fft.fft2(exit_waves, axes=(-2, -1)))

        intensities = np.abs(fft_waves) ** 2

        # Add noise
        if poisson_noise:
            intensities = np.random.poisson(intensities)
        return intensities
    

    def visualize_data(self, rotate: bool = False) -> None:
        """
        Visualize FFT intensities, phases, and amplitudes of the exit waves, plus
        differences versus a homogeneous medium. Optionally uses the precomputed
        rotated forward model (if available).
        """
        self.visualisation = Visualisation(self.simulation_space, results_dir=self.results_dir)
        # Select exit waves according to the orientation
        if rotate and (self.true_exit_waves_rot is not None):
            exit_waves = self.true_exit_waves_rot
            n_for_homog_shape = self.n_true_rot.shape
            title_prefix = "(rotated)"

            n_true = self.n_true_rot
        else:
            if rotate:
                self._log("Warning: rotate=True requested, but rotated forward model "
                      "was not precomputed (requires nx == nz). Using non-rotated data.")
            exit_waves = self.true_exit_waves
            n_for_homog_shape = self.n_true.shape
            title_prefix = ""
            n_true = self.n_true


        self._log("Plot True Object")
        self.visualisation.plot_single(n_true, view="phase_amp", time="final",
                                       filename=f"{'rot_' if title_prefix else ''}true_object.png")

        # Compute homogeneous forward solution (same shape/orientation as selected case)
        n_homogeneous = np.ones(n_for_homog_shape, dtype=complex) * self.simulation_space.n_medium
        if title_prefix:
            # If we're visualizing the rotated case, make sure we pass the rotated
            # n to the forward model (keep other settings identical).
            u_homogeneous = self.convert_to_block_form(self.forward_model.solve(n=n_homogeneous))
        else:
            u_homogeneous = self.convert_to_block_form(self.forward_model.solve(n=n_homogeneous))

        exit_waves_homogeneous = u_homogeneous[:, -self.block_size:]
        diff_exit_waves = exit_waves_homogeneous - exit_waves

        # ---------- FFT-squared intensities ----------
        data = np.zeros((self.num_probes * self.num_angles, self.block_size))
        diff_data = np.zeros_like(data)

        for i in range(self.num_probes * self.num_angles):
            data[i, :] = np.square(np.abs(np.fft.fft(exit_waves[i, :])))

            if self.poisson_noise:
                data[i, :] = np.random.poisson(data[i, :])

            diff_exit_wave_fft = np.fft.fft(diff_exit_waves[i, :])
            diff_data[i, :] = np.square(np.abs(diff_exit_wave_fft))

        if rotate:
            self.data_rot = data
        else:
            self.data = data

        if self._results_dir:
            fig = plt.figure(figsize=(8, 4))
            plt.imshow(data, cmap='viridis', origin='lower')
            plt.colorbar(label='Intensity')
            plt.title(f'Exit Wave Squared FFT Intensity {title_prefix}'.strip())
            plt.xlabel('x'); plt.ylabel('Image #'); plt.tight_layout()
            fig.savefig(os.path.join(self._results_dir, f'true_fft_intensity{ "_rot" if title_prefix else ""}.png'),
                        bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(8, 4))
            plt.imshow(diff_data, cmap='viridis', origin='lower')
            plt.colorbar(label='Intensity')
            plt.title(f'Differences in Exit Waves {title_prefix}:\nFar Field Intensity'.strip())
            plt.xlabel('x'); plt.ylabel('Image #'); plt.tight_layout()
            fig.savefig(os.path.join(self._results_dir, f'true_fft_intensity_diff{ "_rot" if title_prefix else ""}.png'),
                        bbox_inches="tight")
            plt.close(fig)

        # ---------- Phase & Amplitude (and differences) ----------
        self.visualisation.plot_single(
                exit_waves, view="phase_amp", time="final",
                filename=f"exit_phase_amp{ '_rot' if title_prefix else ''}.png",
                title_left=f"Exit Wave Phase {title_prefix}".strip(),
                title_right=f"Exit Wave Amplitude {title_prefix}".strip(),
                xlabel_left="x",  ylabel_left="Image #",
                xlabel_right="x", ylabel_right="Image #",
            )

        self.visualisation.plot_single(
                diff_exit_waves, view="phase_amp", time="final",
                filename=f"exit_phase_amp_diff{ '_rot' if title_prefix else ''}.png",
                title_left=f"Phase Differences in Exit Waves {title_prefix}:\nHomogeneous vs. Inhomogeneous Media".strip(),
                title_right=f"Amplitude Differences in Exit Waves {title_prefix}:\nHomogeneous vs. Inhomogeneous Media".strip(),
                xlabel_left="x",  ylabel_left="Image #",
                xlabel_right="x", ylabel_right="Image #",
            )