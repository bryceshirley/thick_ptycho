import numpy as np
from typing import Optional
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
        sample_space: SampleSpace,
        data: np.ndarray,
        data_is_intensity: bool,
        results_dir: Optional[str] = None,
        use_logging: bool = True,
        verbose: bool = False,
        use_gs_init: bool = False,
        gs_iters: int = 10,
        log_file_name: Optional[str] = None,
    ):
        self.sample_space = sample_space
        self.data = np.asarray(data)
        self.data_is_intensity = data_is_intensity
        self.verbose = verbose
        self.use_gs_init = use_gs_init
        self.gs_iters = gs_iters

        # Logging setup
        log_name = log_file_name or f"{self.__class__.__name__.lower()}_log.txt"
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

    def _gerchberg_saxton_init(self, intensities: np.ndarray, iters: int = 10) -> np.ndarray:
        """
        Perform Gerchberg–Saxton (GS) phase retrieval to estimate complex exit waves
        from intensity-only data.

        Parameters
        ----------
        intensities : np.ndarray
            Far-field intensities (|FFT(exit)|²).
        iters : int, default=10
            Number of GS iterations.

        Returns
        -------
        np.ndarray
            Complex-valued exit waves with recovered phase.
        """
        sqrtI = np.sqrt(intensities)
        phi = np.exp(1j * 2 * np.pi * np.random.rand(*sqrtI.shape))
        est = sqrtI * phi

        for _ in range(iters):
            psi = np.fft.ifft(est)
            est = sqrtI * np.exp(1j * np.angle(np.fft.fft(psi)))

        return np.fft.ifft(est)

    # -------------------------------------------------------------------------
    # Abstract interface
    # -------------------------------------------------------------------------

    def reconstruct(self, *args, **kwargs):
        """Subclasses must implement their reconstruction logic here."""
        raise NotImplementedError("Subclasses must implement the 'reconstruct()' or 'solve()' method.")
