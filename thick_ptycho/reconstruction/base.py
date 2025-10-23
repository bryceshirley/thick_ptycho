import numpy as np
from typing import Optional
from thick_ptycho.thick_ptycho.simulation import ptycho_object, simulation_space
from thick_ptycho.thick_ptycho.simulation.ptycho_object import SampleSpace
from thick_ptycho.utils.utils import setup_log


class ReconstructorBase:
    """
    Base class for all reconstruction algorithms in thick_ptycho.

    Provides common functionality such as:
    - Data validation
    - Logging and verbosity control
    - Optional Gerchberg–Saxton (GS) initialization
    - Common data handling for intensity or complex measurements

    Subclasses must implement their specific reconstruction logic,
    typically via a `solve()` or `reconstruct()` method.

    Parameters
    ----------
    sample_space : SampleSpace
        Object defining geometry, grid spacing, and refractive index ranges.
    data : np.ndarray
        Measurement data (intensity or complex exit waves).
    data_is_intensity : bool
        Whether `data` represents intensities (|FFT(exit)|²) or complex fields.
    results_dir : str, optional
        Directory path for log files (no plots generated).
    use_logging : bool, default=True
        Enable internal logging.
    verbose : bool, default=False
        Print progress messages to stdout.
    use_gs_init : bool, default=False
        Perform Gerchberg–Saxton initialization if data is intensity-only.
    gs_iters : int, default=10
        Number of GS iterations for phase initialization.
    log_file_name : str, optional
        Name of the log file. Defaults to "<classname>_log.txt".
    """

    def __init__(
        self,
        simulation_space: simulation_space,
        ptycho_object: ptycho_object,
        ptycho_probes: np.ndarray,
        data,
        phase_retrieval: bool = False,
        results_dir=None,
        use_logging=True,
        verbose=False,
        **kwargs,
    ):
        self.simulation_space = simulation_space
        self.ptycho_object = ptycho_object
        self.ptycho_probes = ptycho_probes
        self.data = np.asarray(data)
        self.phase_retrieval = phase_retrieval
        self.verbose = verbose

        self.num_angles = self.simulation_space.num_angles
        self.num_probes = self.simulation_space.num_probes

        # Logging setup
        log_name = f"{self.__class__.__name__.lower()}_log.txt"
        self._log = setup_log(
            results_dir,
            log_file_name=log_name,
            use_logging=use_logging,
            verbose=verbose,
        )

        # Data validation
        if data_is_intensity and np.iscomplexobj(self.data):
            raise ValueError("Intensity data must be real-valued.")
        if not data_is_intensity and not np.iscomplexobj(self.data):
            raise ValueError("Complex exit wave data expected for data_is_intensity=False.")

        # Optional GS initialization
        if self.data_is_intensity and self.use_gs_init:
            self._log(f"Performing Gerchberg–Saxton initialization for {self.gs_iters} iterations...")
            self.data = self._gerchberg_saxton_init(self.data, self.gs_iters)
            self.data_is_intensity = False
            self._log("Gerchberg–Saxton initialization complete.")

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def apply_exit_wave_constraint(self, uk):
        """
        Compute the error in the exit wave.

        Parameters
        ----------
        uk : np.ndarray
            Current solution vector, shape (num_probes * num_angles*rotation_angles, nx * (nz - 1)).
        known_phase : bool, optional
            If True, compares directly to the true exit wave.
            If False, performs a phase retrieval constraint update.

        Returns
        -------
        np.ndarray
            The complex-valued exit wave error, same shape as uk.
        """
        exit_wave_error = np.zeros_like(uk, dtype=complex)
        exit_waves = uk[:, -self.block_size:]

        if self.phase_retrieval:
            emodel = self._apply_phase_retrieval_constraint(exit_waves)
            exit_wave_error[:, -self.block_size:] = exit_waves - emodel

            if getattr(self, "_results_dir", None):
                # Optional visualization of pre/post phase retrieval
                self.visualisation.plot_single(
                    exit_waves, view="phase_amp", time="final",
                    filename="exit_phase_amp_old.png",
                    title_left="Old Exit Wave Phase",
                    title_right="Old Exit Wave Amplitude",
                    xlabel_left="x", ylabel_left="Image #",
                    xlabel_right="x", ylabel_right="Image #",
                )

                self.visualisation.plot_single(
                    emodel, view="phase_amp", time="final",
                    filename="exit_phase_amp_updated.png",
                    title_left="Updated Exit Wave Phase",
                    title_right="Updated Exit Wave Amplitude",
                    xlabel_left="x", ylabel_left="Image #",
                    xlabel_right="x", ylabel_right="Image #",
                )

        else:  # Known Phase
            exit_wave_error[:, -self.block_size:] = exit_waves - self.data

        return exit_wave_error


    def _apply_phase_retrieval_constraint(self, exit_waves: np.ndarray) -> np.ndarray:
        """
        Apply a Gerchberg–Saxton-like phase retrieval constraint in Fourier space.

        This replaces the amplitude in the Fourier domain with the measured data's
        amplitude while retaining the current phase estimate, and then transforms back.

        Parameters
        ----------
        exit_waves : np.ndarray
            Exit wave field for each probe/angle, shape (num_probes * num_angles, block_size).
        rotate : bool, optional
            Whether to use rotated data (default: False).

        Returns
        -------
        np.ndarray
            Updated exit waves after applying the phase retrieval constraint.
        """
        # Forward FFT
        fmodel = np.fft.fft(exit_waves, axis=-1)

        # Avoid division by zero
        magnitude = np.abs(fmodel)
        magnitude[magnitude == 0] = 1.0
        phase = fmodel / magnitude

        # Select measured amplitude data
        target_amplitude = np.sqrt(self.data)

        # Replace amplitude while keeping phase (vectorized)
        fmodel_updated = target_amplitude * phase

        # Inverse FFT to get updated exit waves
        emodel = np.fft.ifft(fmodel_updated, axis=-1)

        return emodel

    # -------------------------------------------------------------------------
    # Abstract interface
    # -------------------------------------------------------------------------

    def reconstruct(self, *args, **kwargs):
        """Subclasses must implement their reconstruction logic here."""
        raise NotImplementedError("Subclasses must implement the 'reconstruct()' or 'solve()' method.")
