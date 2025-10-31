import numpy as np
from typing import Optional
# from thick_ptycho.simulation import ptycho_object, simulation_space
from thick_ptycho.thick_ptycho.utils.io import setup_log
from thick_ptycho.simulation.ptycho_object import create_ptycho_object
from thick_ptycho.simulation.ptycho_probe import create_ptycho_probes


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
    log_file_name : str, optional
        Name of the log file. Defaults to "<classname>_log.txt".
    """

    def __init__(
        self,
        simulation_space,
        data,
        phase_retrieval: bool = False,
        results_dir=None,
        use_logging=True,
        verbose=False,
        **kwargs,
    ):
        self.simulation_space = simulation_space
        self.ptycho_object = create_ptycho_object(simulation_space)
        self.ptycho_probes = create_ptycho_probes(simulation_space)
        self.data = np.asarray(data)
        #assert len(self.data.shape) == 4, "Data must be a 4D array (projections, angles, probes, pixels)."
        self.phase_retrieval = phase_retrieval
        self.verbose = verbose
        self._results_dir = results_dir

        self.num_angles = self.simulation_space.num_angles
        self.num_probes = self.simulation_space.num_probes
        self.num_projections = self.simulation_space.num_projections
        self.total_scans = self.num_angles * self.num_probes * self.num_projections

        # Logging setup
        log_name = f"{self.__class__.__name__.lower()}_log.txt"
        self._log = setup_log(
            results_dir,
            log_file_name=log_name,
            use_logging=use_logging,
            verbose=verbose,
        )

        self.block_size = self.simulation_space.block_size
        self.nx = simulation_space.nx
        self.nz = simulation_space.nz

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def apply_exit_wave_constraint(self, uk):
        """
        Compute the error in the exit wave.

        Parameters
        ----------
        uk : np.ndarray
            Current solution vector, shape (total_scans, nx * (nz - 1)).
        known_phase : bool, optional
            If True, compares directly to the true exit wave.
            If False, performs a phase retrieval constraint update.

        Returns
        -------
        np.ndarray
            The complex-valued exit wave error, same shape as uk.
        """
        #exit_waves = uk[:, -self.block_size:]
        exit_wave_error = np.zeros_like(self.convert_to_tensor_form(uk), dtype=complex)
        exit_waves = self.convert_to_tensor_form(uk)[...,-1].reshape(self.total_scans, 
                                                                     self.block_size)

        if self.phase_retrieval:
            emodel = self._apply_phase_retrieval_constraint(exit_waves)
            exit_wave_error[..., :, -1] = (exit_waves - emodel).reshape(self.num_projections,
                                                                           self.num_angles,
                                                                           self.num_probes,
                                                                             self.block_size)


            if getattr(self, "_results_dir", None):
                # Optional visualization of pre/post phase retrieval
                self.simulation_space.viewer.plot_two_panels(
                    exit_waves, view="phase_amp",
                    filename="exit_phase_amp_old.png",
                    title="Old Exit Wave",
                    xlabel="x", ylabel="Image #"
                )

        else:  # Known Phase
            exit_wave_error[..., :, -1] = (exit_waves - self.data).reshape(self.num_projections,
                                                                           self.num_angles,
                                                                           self.num_probes,
                                                                             self.block_size)

        return exit_wave_error
    
    def convert_to_tensor_form(self,u):
        """
        Reverse the block flattening process.

        Parameters:
        u (ndarray): Flattened array of shape (num_angles, num_probes, block_size * (nz - 1))

        Returns:
        ndarray: Unflattened array of shape (num_angles, num_probes, block_size, nz - 1)
        """
        # Step 1: Reshape to (num_probes, nz - 1, block_size)
        reshaped = u.reshape(self.num_projections,self.num_angles, self.num_probes, self.nz-1, self.block_size)

        # Step 2: Transpose to (num_probes, num_probes, nx, nz - 1)
        return reshaped.transpose(0, 1, 2, 4, 3)

    def convert_to_vector_form(self, u):
        """
        Convert the input array to block form.

        Parameters:
        u (ndarray): Input array to be converted. shape: (projection_number, num_angles, num_probes, nx, nz)

        Returns:
        ndarray: Block-formatted array. (total_scans, nx*nz)
        """
        # 2. Remove initial condition
        u = u[:, :, :, :, 1:]  # shape: (projection_number, num_angles, num_probes, block_size, nz - 1)

        # 3. Transpose axes
        u = u.transpose(0, 1, 2, 4, 3) # shape: (projection_number, num_angles, num_probes, nz - 1, block_size)

        # 4. Flatten last two dims
        # shape: (total_scans, block_size * (nz - 1))
        u = u.reshape(self.total_scans, -1)

        return u


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
