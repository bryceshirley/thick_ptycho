import numpy as np
import time
from typing import Optional, List, Literal, Any

from thick_ptycho.sample_space.sample_space import SampleSpace
from thick_ptycho.utils.utils import setup_log
from thick_ptycho.forward_model.probes import Probes

# # ---------------------------- Enums & Config ---------------------------- #

# class SolverType(Enum):
#     """Algorithmic strategy for solving the paraxial wave equation."""
#     TIME_STEPPING = auto()    # formerly "IterativeSolver"
#     BLOCK_MATRIX = auto()     # formerly "FullSystemSolver"
#     MULTISLICE = auto()       # new: standard multislice propagation


# class PropagationMode(Enum):
#     """Direction of propagation."""
#     FORWARD = auto()
#     ADJOINT = auto()
#     REVERSE = auto()  # (aka back-propagation / adjoint-like transport)


# @dataclass(frozen=True)
# class SolverConfig:
#     """High-level configuration for the solver orchestration."""
#     solver_type: SolverType = SolverType.TIME_STEPPING
#     propagation: PropagationMode = PropagationMode.FORWARD
#     test_impedance: bool = False
#     thin_sample: bool = True
#     verbose: bool = True
#     use_logging: bool = False
#     results_dir: str = ""



import numpy as np
import time
from typing import Optional, List, Literal, Any, Dict

from thick_ptycho.sample_space.sample_space import SampleSpace
from thick_ptycho.utils.utils import setup_log
from thick_ptycho.forward_model.probes import Probes


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
        sample_space: SampleSpace,
        solver_type: Literal["pwe_iterative", "pwe_full", "multislice"],
        thin_sample: bool = True,
        probe_angles_list: Optional[List[Any]] = None,
        results_dir: str = "",
        use_logging: bool = False,
        verbose: bool = True,
        log=None,
    ):
        self.sample_space = sample_space
        self.solver_type = solver_type.lower()
        self.thin_sample = thin_sample
        self.nz = sample_space.nz
        self.verbose = verbose
        self.num_probes = sample_space.num_probes
        self.probe_angles_list = probe_angles_list or [0.0]

        # Logger
        self._log = log or setup_log(results_dir, "solver_log.txt", use_logging, verbose)

        # Determine slice dimensions
        if sample_space.dimension == 1:
            nx = sample_space.sub_nx if thin_sample else sample_space.nx
            self.slice_dimensions = (nx,)
        elif sample_space.dimension == 2:
            nx = sample_space.sub_nx if thin_sample else sample_space.nx
            ny = sample_space.sub_ny if thin_sample else sample_space.ny
            self.slice_dimensions = (nx, ny)
        else:
            raise ValueError("Unsupported sample space dimension")

        # Probe setup
        self.probe_builder = Probes(sample_space, thin_sample, angles_list=self.probe_angles_list)
        self.probes = self.probe_builder.build_probes()
        self.num_angles = len(self.probe_angles_list)

    # ------------------------------------------------------------------
    # Common solving interface
    # ------------------------------------------------------------------

    def solve(self, n: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Main multi-angle, multi-probe solving loop.
        Subclasses must define `_solve_single_probe(angle_idx, probe_idx, n, **kwargs)`.
        """
        u = self._create_solution_grid()
        for a_idx, angle in enumerate(self.probe_angles_list):
            for p_idx in range(self.num_probes):
                start = time.time()
                u[a_idx, p_idx, ...] = self._solve_single_probe(
                    angle_idx=a_idx, probe_idx=p_idx, n=n, **kwargs
                )
                if self.verbose:
                    self._log(
                        f"[{self.solver_type}] solved probe {p_idx+1}/{self.num_probes} "
                        f"at angle {angle} in {time.time() - start:.2f}s"
                    )
        return u

    def _solve_single_probe(self, angle_idx: int, probe_idx: int, n=None):
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
        if self.sample_space.dimension == 1:
            return u[..., -1]
        elif self.sample_space.dimension == 2:
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
        exit_waves = exit_waves or self.simulate_exit_waves(n=n, **kwargs)

        # Compute FFTs for all probes and angles
        if self.sample_space.dimension == 1:
            fft_waves = np.fft.fftshift(np.fft.fft(exit_waves, axis=-1))
        else:
            fft_waves = np.fft.fftshift(np.fft.fft2(exit_waves, axes=(-2, -1)))

        intensities = np.abs(fft_waves) ** 2

        # Add noise
        if poisson_noise:
            intensities = np.random.poisson(intensities)
        return intensities




# import time
# from typing import Optional, Tuple, Any, List

# import numpy as np
# import scipy.sparse as sp
# import scipy.sparse.linalg as spla

# from .PWE.paraxial_wave_equation import LinearSystemSetup
# from thick_ptycho.sample_space.sample_space import SampleSpace
# from thick_ptycho.utils.utils import setup_log
# from thick_ptycho.forward_model.probes import Probes


# class ForwardModel():
#     """
#     A solver for the paraxial wave equation using finite difference methods for ptychography.
#     Handles various initial conditions, boundary conditions, and objects in free space.
#     """

#     def __init__(self, sample_space: SampleSpace,
#                  full_system_solver: Optional[bool] = False,
#                  thin_sample: Optional[bool] = True,
#                  probe_angles_list: Optional[List[float]] = [0.0],
#                  log = None,
#                  results_dir: str = "",
#                  use_logging: Optional[bool] = False,
#                  verbose: Optional[bool] = True
#                  ):
#         """
#         Initialize the solver.
#         """
#         if log:
#             self._log = log
#         else:
#             self._log = setup_log(results_dir,log_file_name="solver_log.txt",
#                                use_logging=use_logging, verbose=verbose)

#         self.sample_space = sample_space
#         self.nz = sample_space.nz
#         self.probe_dimensions = sample_space.probe_dimensions
#         self.thin_sample = thin_sample
#         self.full_system = full_system_solver

#         # Determine slice dimensions based on sample type and boundary conditions
#         if self.sample_space.dimension == 2:
#             nx = self.sample_space.sub_nx if self.thin_sample else self.sample_space.nx
#             ny = self.sample_space.sub_ny if self.thin_sample else self.sample_space.ny
#             self.slice_dimensions = (nx, ny)
#         elif self.sample_space.dimension == 1:
#             nx = self.sample_space.sub_nx if self.thin_sample else self.sample_space.nx
#             self.slice_dimensions = (nx,)
#         else:
#             raise ValueError("Unsupported sample_space dimension")

#         # if self.sample_space.bc_type == "dirichlet":
#         #     self.slice_dimensions_dirichlet = tuple(dim + 2 for dim in self.slice_dimensions)

#         # Probe generator
#         self.probe_angles_list = probe_angles_list
#         self.probe_builder = Probes(self.sample_space, thin_sample=thin_sample,
#                                     probe_angles_list=probe_angles_list) # (num_probes, nx)

#         # Precompute probes for all scans × angles
#         self.probes = self.probe_builder.build_probes()
#         self.num_probes = len(self.probes)


#         self.linear_system = ForwardModelPWE(sample_space,
#                                                self.thin_sample,
#                                                self.full_system)

#         self.block_size = self.linear_system.block_size

#     def solve(self, reverse=False,
#               initial_condition: Optional[np.ndarray] = None,
#               test_impedance=False,
#               verbose: Optional[bool] = False,
#               n: Optional[np.ndarray] = None,
#               lu: Optional[spla.SuperLU] = None,
#               iterative_lu: Tuple[Optional[list[spla.SuperLU]], Optional[list[np.ndarray]], Optional[list[np.ndarray]]] = None,
#               b_block: Optional[list[np.ndarray]] = None) -> np.ndarray:
#         """Solve the paraxial wave equation."""
#         # Initialize solution grid with initial condition
#         u = self._create_solution()

#         if not self.thin_sample and self.full_system:
#             if lu is None:
#                 lu = spla.splu(self.return_forward_model_matrix(n=n))
#             b_homogeneous = self.linear_system.setup_homogeneous_forward_model_rhs()
#         else:
#             b_homogeneous = None
        
#         if not self.thin_sample and not self.full_system:
#             if iterative_lu is None:
#                 iterative_lu = self.construct_iterative_lu(
#                     n=n, reverse=reverse)

#         # Define wrapper for parallel execution
#         def solve_single_probe(scan_index: int, angle_index: float) -> np.ndarray:
#             """Solve the paraxial wave equation for a single probe."""

#             start_time = time.time()
#             if self.full_system:
#                 sol = self._solve_single_probe_full_system(
#                     n, scan_index=scan_index, angle_index=angle_index, reverse=reverse,
#                     initial_condition=initial_condition,
#                     test_impedance=test_impedance,
#                     lu=lu,
#                     b_homogeneous=b_homogeneous)
#             else:
#                 sol = self._solve_single_probe_iteratively(
#                     n, scan_index=scan_index, angle_index=angle_index, reverse=reverse,
#                     initial_condition=initial_condition,
#                     test_impedance=test_impedance,
#                     iterative_lu=iterative_lu,
#                     b_block=b_block)
#             # # Handle Dirichlet BCs
#             # sol = self._handle_dirichlet_bcs(sol)
#             end_time = time.time()

#             # Time taken for each scan if verbose is True
#             if verbose:
#                 self._log(f"Time to solve scan {scan_index+1}/{self.sample_space.num_probes}: {end_time - start_time} seconds")
            
#             return sol
        
#         # This could be made parallel
#         for angle_index, probe_angle in enumerate(self.probe_angles_list):
#             if verbose and len(self.probe_angles_list) > 1:
#                 self._log(f"Solving for probe angle {angle_index+1}/{len(self.probe_angles_list)}: {probe_angle} radians")
#             for scan_index in range(self.sample_space.num_probes):
#                 u[angle_index,scan_index, ...] = solve_single_probe(scan_index, angle_index=angle_index)

#         return u

#     def construct_iterative_lu(self, n: Optional[np.ndarray] = None, adjoint: bool = False,
#                             reverse: bool = False) -> Tuple[Optional[spla.SuperLU], Optional[np.ndarray]]:
#         """Solve the paraxial wave equation iteratively for a single probe."""
#         # Precompute C, A_mod, B_mod, LU factorizations
#         A_lu_list = []
#         B_mod_list = []

#         object_slices = self.sample_space.create_sample_slices(
#                             self.thin_sample,
#                             n=n).reshape(-1, self.nz - 1)


#         # Create Linear System and Apply Boundary Conditions
#         A, B, b = self.linear_system.create_system_slice()

#         if reverse:
#             A, B = B, A
#             b = - b

#         # Iterate over the z dimension
#         for j in range(1, self.nz):
#             if reverse:
#                 C = - sp.diags(object_slices[:, -j])
#             elif adjoint:
#                 C = sp.diags(object_slices[:, -j])
#             else:
#                 C = sp.diags(object_slices[:, j - 1])

#             A_with_object = A - C  # LHS Matrix
#             B_with_object = B + C  # RHS Matrix

#             if adjoint:
#                 A_with_object, B_with_object = A_with_object.conj().T, B_with_object.conj().T

#             A_lu = spla.splu(A_with_object.tocsc())
#             A_lu_list.append(A_lu)
#             B_mod_list.append(B_with_object)
        

#         return (A_lu_list, B_mod_list, b)

#     def _solve_single_probe_full_system(self, n: np.ndarray,
#                                         angle_index: int = 0,
#                                         scan_index: int = 0,
#                                         reverse: bool = False,
#                                         initial_condition: Optional[np.ndarray] = None,
#                                         test_impedance: bool = False,
#                                         lu: Optional[spla.SuperLU] = None,
#                                         b_homogeneous: Optional[np.ndarray] = None) -> np.ndarray:
#         """Solve the paraxial wave equation for a single probe using the full system."""
#         if reverse:
#             raise ValueError(
#                 "Reverse propagation is not supported in the full system solver. "
#                 "Please use the iterative solver for reverse propagation.")
    
#         # Forward model rhs vector
#         if test_impedance:
#             b_homogeneous = (
#                 self.linear_system.test_exact_impedance_forward_model_rhs()
#             )
#         elif b_homogeneous is None:
#             b_homogeneous = (
#                 self.linear_system.setup_homogeneous_forward_model_rhs(
#                     scan_index=scan_index)
#             )

#         # Edit this for impedance condition test
#         probe_contribution = self.linear_system.probe_contribution(
#             scan_index=scan_index,angle_index=angle_index, probes=initial_condition)

#         # Define Right Hand Side
#         b = b_homogeneous + probe_contribution

#         # Solve Forward Model and Restrict to Detector 
#         if lu is not None:
#             solution = lu.solve(b)
#         else:
#             # Forward Model lhs matrix
#             forward_model_matrix = self.return_forward_model_matrix(
#                 scan_index=scan_index, n=n)
#             solution = spla.spsolve(forward_model_matrix, b)

#         # Reshape solution
#         solution = solution.reshape(self.nz - 1,
#                                     self.block_size).transpose()

#         # Concatenate initial condition (nx, 1) with solution (nx, nz-1)
#         initial_condition = probe.reshape(self.block_size, 1)
#         return np.concatenate([initial_condition, solution], axis=1)

#     def _solve_single_probe_iteratively(self, n: Optional[np.ndarray] = None,
#                                         angle_index: Optional[int] = 0,
#                                         scan_index: Optional[int] = 0,
#                                         reverse: Optional[bool] = False,
#                                         adjoint: Optional[bool] = False,
#                                         initial_condition: Optional[np.ndarray] = None,
#                                         test_impedance: Optional[bool] = False,
#                                         iterative_lu: Optional[Tuple[list[spla.SuperLU], list[np.ndarray], list[np.ndarray]]] = None,
#                                         b_block: Optional[list[np.ndarray]] = None) -> np.ndarray:
#         """Solve the paraxial wave equation iteratively for a single probe."""
#         # Initialize solution grid
#         solution = np.zeros((self.block_size, self.nz),
#                             dtype=complex)
#         probe_index = angle_index*self.sample_space.num_probes + scan_index

#         # Set initial condition
#         if isinstance(initial_condition, int):
#             pass
#         else:
#             if reverse:
#                 if initial_condition is None:
#                     raise ValueError("Exit wave required for reverse propagation.")
#                 # initial_condition expected shape (num_probes, block_size)
#                 solution[:, 0] = initial_condition[probe_index, :].flatten()
#             elif initial_condition is None:
#                 solution[:, 0] = self.probes[probe_index, ...].flatten()
#             else:
#                 solution[:, 0] = initial_condition[probe_index, ...].flatten()

#         # Solve with LU decomposition
#         if iterative_lu is not None:
#             A_lu_list, B_list, b = iterative_lu

#             if b_block is not None:
#                 b_block = b_block.reshape(self.nz-1, self.block_size).transpose()
#             # Iterate over the z dimension
#             for j in range(1, self.nz):
#                 if test_impedance:
#                     b = self.linear_system.test_exact_impedance_rhs_slice(j)
#                 if b_block is not None:
#                     if adjoint:
#                         b = b_block[:, -j]
#                     else:
#                         b = b_block[:, j-1]

#                 rhs_matrix = B_list[j - 1].dot(solution[:, j - 1]) + b
#                 solution[:, j] = A_lu_list[j - 1].solve(rhs_matrix)#*self.sponge_mask()
#         # For thin samples with spsolve
#         else:
#             object_slices = (
#                     self.sample_space.create_sample_slices(
#                         self.thin_sample, n=n, scan_index=scan_index
#                     )
#                 ).reshape(-1, self.nz - 1)

#             # Create Linear System and Apply Boundary Conditions
#             A, B, b = self.linear_system.create_system_slice(
#                 scan_index=scan_index)

#             if reverse:
#                 A, B = B, A
#                 b = - b

#             # Iterate over the z dimension
#             for j in range(1, self.nz):
#                 if test_impedance:
#                     b = self.linear_system.test_exact_impedance_rhs_slice(j)
                
#                 if reverse:
#                     C = - sp.diags(object_slices[:, -j])
#                 else:
#                     C = sp.diags(object_slices[:, j - 1])

#                 A_with_object = A - C  # LHS Matrix
#                 B_with_object = B + C  # RHS Matrix
#                 rhs_matrix = B_with_object.dot(solution[:, j - 1]) + b
#                 solution[:, j] = spla.spsolve(A_with_object, rhs_matrix)

#         # Flip solution in the z-direction if adjoint is True
#         if adjoint:
#             solution = np.flip(solution, axis=1)
#         return solution

#     def return_forward_model_matrix(self, scan_index: int = 0,
#                                     n: np.ndarray = None) -> np.ndarray:
#         """Return the forward model matrix for a given scan index."""
#         # Create the inhomogeneous forward model matrix
#         A_homogeneous = (
#             self.linear_system.setup_homogeneous_forward_model_lhs(
#                 scan_index=scan_index)
#         )

#         Ck = self.linear_system.setup_inhomogeneous_forward_model(
#             n=n, scan_index=scan_index)

#         return (A_homogeneous - Ck).tocsc()  # Convert to Compressed Sparse Column format for efficiency

#     def _create_solution(self) -> np.ndarray:
#         """Create the solution grid based on the sample space and boundary 
#         conditions."""
#         return np.zeros(
#             (len(self.probe_angles_list), self.sample_space.num_probes,
#                 *self.slice_dimensions, self.nz),
#             dtype=complex
#         )


# class ForwardModelMultiSlice:
#     """
#     1D Angular Spectrum Multi-slice Forward and Inverse (3PIE-style) Model.
#     Implements forward angular spectrum propagation and inverse reconstruction
#     per Maiden, Humphry, and Rodenburg (2012).
#     """

#     def __init__(self,
#                  sample_space,
#                  pad_factor: float = 1.0,
#                  use_padding: bool = False,
#                  dtype=np.complex64,
#                  remove_global_phase: bool = True,
#                  normalize_probes: bool = True):

#         assert sample_space.dimension == 1, "MultiSliceForwardModel expects 1D SampleSpace."

#         self.sample_space = sample_space
#         self.k = sample_space.k
#         self.wavelength = 2 * np.pi / self.k
#         self.nx = sample_space.nx
#         self.nz = sample_space.nz
#         self.dx = sample_space.dx
#         self.dz = sample_space.dz
#         self.n_medium = sample_space.n_medium
#         self.dtype = dtype
#         self.remove_global_phase = remove_global_phase
#         self.normalize_probes = normalize_probes

#         # --- Object field
#         self.n_field = sample_space.n_true  # (nx, nz)

#         # --- Detector info
#         self._detector_info = sample_space.detector_frame_info
#         self._probe_half_width = sample_space.probe_dimensions[0] // 2

#         # Probe generator
#         self.probe_builder = Probes(self.sample_space, thin_sample=False)[0] # (num_probes, nx)

#         # Precompute probes for all scans × angles
#         self.probes = self.probe_builder.build_probes()
#         self.num_probes = len(self.probes)

#         # --- Minimal cache (disabled unless needed)
#         self.use_padding = use_padding
#         self.pad_factor = pad_factor
#         self._kernel_cache = {}

#     # --------------------------------------------------------------------------
#     # Angular Spectrum Propagation
#     # --------------------------------------------------------------------------
#     def _get_kernel(self, dz: float, nx_eff: Optional[int] = None):
#         """Angular spectrum kernel for forward propagation."""
#         nx_eff = nx_eff or self.nx
#         key = (dz, nx_eff)
#         if key in self._kernel_cache:
#             return self._kernel_cache[key]

#         fx = np.fft.fftfreq(nx_eff, d=self.dx)
#         kx = 2 * np.pi * fx
#         inside = (self.k ** 2 - kx ** 2)
#         kz = np.sqrt(np.clip(inside, 0.0, None))
#         H = np.exp(1j * kz * dz)
#         if self.remove_global_phase:
#             H *= np.exp(-1j * self.k * dz)

#         self._kernel_cache[key] = H.astype(self.dtype)
#         return self._kernel_cache[key]

#     def _propagate(self, psi, dz):
#         """Forward propagation through background medium."""
#         H = self._get_kernel(dz, psi.size)
#         Psi = np.fft.fft(psi)
#         Psi *= H
#         return np.fft.ifft(Psi)

#     def _backpropagate(self, psi, dz):
#         """Inverse propagation (negative dz)."""
#         return self._propagate(psi, -dz)

#     def forward(self, n: Optional[np.ndarray] = None, probe_idx: Optional[int] = None):
#         """
#         Forward angular spectrum multislice propagation.

#         Returns
#         -------
#         exit_fields : np.ndarray
#             Complex detector-plane field for each probe (num_probes, nx).
#         psi_slices_all : list of list of np.ndarray
#             psi_slices_all[p][z] gives the complex field for probe p at slice index z.
#             Shape of psi_slices_all[p][z] = (nx,)
#         probe_stack : np.ndarray
#             Stack of input probes (num_probes, nx).
#         """
#         n = n if n is not None else self.n_field
#         assert n.shape == (self.nx, self.nz)

#         # Choose subset of probes
#         probes = self.probes if probe_idx is None else [self.probes[probe_idx]]

#         if probe_idx is not None:
#             num_probes = 1
#         else:
#             num_probes = self.num_probes
#         psi_slices = np.empty((num_probes, self.nx, self.nz), dtype=self.dtype)

#         for p, psi0 in enumerate(probes):
#             psi_i = psi0.copy()
#             psi_slices[p, :, 0] = psi_i.copy()  # field at first slice plane

#             # Propagate through all slices
#             for z in range(self.nz - 1):
#                 Tz = np.exp(1j * self.k * (n[:, z] - self.n_medium) * self.dz)
#                 psi_i = self._propagate(psi_i * Tz, self.dz)
#                 psi_slices[p, :, z + 1] = psi_i.copy()

#         return psi_slices


#     @staticmethod
#     def _update_object(nz, Sz, dpsi, alpha, eps=1e-8):
#         # PIE-style normalized gradient step using sensitivity Sz
#         denom = np.max(np.abs(Sz)**2) + eps
#         return nz + alpha * np.conj(Sz) / denom * dpsi
    
#     def _apply_modulus_constraint(self, Psi, meas_amp, gamma=1.0):
#     # gamma in (0, 1] for relaxation; 1.0 = hard constraint
#         amp = np.abs(Psi) + 1e-12
#         Psi_target = meas_amp * (Psi / amp)
#         return (1 - gamma) * Psi + gamma * Psi_target