import time
from typing import Optional, Tuple, Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .initial_conditions import initial_condition_setup
from .linear_system import LinearSystemSetup
from thickptypy.sample_space.sample_space import SampleSpace

class ForwardModel():
    """
    A solver for the paraxial wave equation using finite difference methods.
    Handles various initial conditions, boundary conditions, and objects in free space.
    """

    def __init__(self, sample_space: SampleSpace,
                 full_system_solver: Optional[bool] = False,
                 thin_sample: Optional[bool] = True):
        """
        Initialize the solver.
        """
        self.sample_space = sample_space
        self.nz = sample_space.nz
        self.probe_dimensions = sample_space.probe_dimensions
        self.initial_condition = initial_condition_setup(sample_space)
        self.thin_sample = thin_sample
        self.full_system = full_system_solver

        # Determine slice dimensions based on sample type and boundary conditions
        if self.sample_space.dimension == 2:
            nx = self.sample_space.sub_nx if self.thin_sample else self.sample_space.nx
            ny = self.sample_space.sub_ny if self.thin_sample else self.sample_space.ny
            self.slice_dimensions = (nx, ny)
        elif self.sample_space.dimension == 1:
            nx = self.sample_space.sub_nx if self.thin_sample else self.sample_space.nx
            self.slice_dimensions = (nx,)
        else:
            raise ValueError("Unsupported sample_space dimension")

        if self.sample_space.bc_type == "dirichlet":
            self.slice_dimensions_dirichlet = tuple(dim + 2 for dim in self.slice_dimensions)

        self.linear_system = LinearSystemSetup(sample_space,
                                               self.initial_condition,
                                               self.thin_sample,
                                               self.full_system)
        
        self.lu = None  # Preconditioner matrix, initialized to None

    def solve(self, reverse=False,
              initial_condition: Optional[np.ndarray] = None,
              test_impedance=False,
              verbose: Optional[bool] = False,
              n: Optional[np.ndarray] = None,
              lu: Optional[spla.SuperLU] = None,
              iterative_lu: Tuple[Optional[list[spla.SuperLU]], Optional[list[np.ndarray]], Optional[list[np.ndarray]]] = None
              ) -> np.ndarray:
        """Solve the paraxial wave equation."""
        # Initialize solution grid with initial condition
        u = self._create_solution()

        if not self.thin_sample and self.full_system:
            if lu is None:
                lu = spla.splu(self.return_forward_model_matrix(n=n))
            b_homogeneous = self.linear_system.setup_homogeneous_forward_model_rhs()
        else:
            b_homogeneous = None
        
        if not self.thin_sample and not self.full_system:
            if iterative_lu is None:
                iterative_lu = self.create_lu_and_b_mod(
                    n=n, reverse=reverse)


        # Define wrapper for parallel execution
        def solve_single_probe(scan_index: int) -> np.ndarray:
            """Solve the paraxial wave equation for a single probe."""

            start_time = time.time()
            if self.full_system:
                sol = self._solve_single_probe_full_system(
                    scan_index, n, reverse=reverse,
                    initial_condition=initial_condition,
                    test_impedance=test_impedance,
                    lu=lu,
                    b_homogeneous=b_homogeneous)
            else:
                sol = self._solve_single_probe_iteratively(
                    scan_index, n, reverse=reverse,
                    initial_condition=initial_condition,
                    test_impedance=test_impedance,
                    iterative_lu=iterative_lu)
            # Handle Dirichlet BCs
            sol = self._handle_dirichlet_bcs(sol)

            # Print time taken for each scan if verbose is True
            if verbose:
                print(f"Time to solve scan {scan_index+1}/{self.sample_space.num_probes}: {end_time - start_time} seconds")
            
            return sol
        
        # This could be made parallel
        for scan_index in range(self.sample_space.num_probes):
            u[scan_index, ...] = solve_single_probe(scan_index)
        
        return u
    
    def create_lu_and_b_mod(self, n: Optional[np.ndarray] = None,
                            reverse: bool = False) -> Tuple[Optional[spla.SuperLU], Optional[np.ndarray]]:
        """Solve the paraxial wave equation iteratively for a single probe."""
        # Precompute C, A_mod, B_mod, LU factorizations
        A_lu_list = []
        B_mod_list = []

        object_slices = self.sample_space.create_sample_slices(
                            self.thin_sample,
                            n=n).reshape(-1, self.nz - 1)


        # Create Linear System and Apply Boundary Conditions
        A, B, b = self.linear_system.create_system_slice()

        if reverse:
            A, B = B, A
            b = - b

        # Iterate over the z dimension
        for j in range(1, self.nz):
            if reverse:
                C = - sp.diags(object_slices[:, -j])
            else:
                C = sp.diags(object_slices[:, j - 1])

            A_with_object = A - C  # LHS Matrix
            B_with_object = B + C  # RHS Matrix

            A_lu = spla.splu(A_with_object.tocsc())
            A_lu_list.append(A_lu)
            B_mod_list.append(B_with_object)

        return (A_lu_list, B_mod_list, b)

    def _solve_single_probe_full_system(self, scan_index: int, n: np.ndarray,
                                        reverse: bool = False,
                                        initial_condition: Optional[np.ndarray] = None,
                                        test_impedance: bool = False,
                                        lu: Optional[spla.SuperLU] = None,
                                        b_homogeneous: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve the paraxial wave equation for a single probe using the full system."""
        if reverse:
            raise ValueError(
                "Reverse propagation is not supported in the full system solver. "
                "Please use the iterative solver for reverse propagation.")
        # Forward model rhs vector
        if test_impedance:
            b_homogeneous = (
                self.linear_system.test_exact_impedance_forward_model_rhs()
            )
        elif b_homogeneous is None:
            b_homogeneous = (
                self.linear_system.setup_homogeneous_forward_model_rhs(
                    scan_index=scan_index)
            )

        # Edit this for impedance condition test
        probe_contribution, probe = self.linear_system.probe_contribution(
            scan_index=scan_index)

        # Define Right Hand Side
        b = b_homogeneous + probe_contribution

        # Solve Forward Model and Restrict to Detector 
        if lu is not None:
            solution = lu.solve(b)
        else:
            # Forward Model lhs matrix
            forward_model_matrix = self.return_forward_model_matrix(
                scan_index=scan_index, n=n)
            solution = spla.spsolve(forward_model_matrix, b)

        # Reshape solution
        solution = solution.reshape(self.nz - 1, 
                                    self.linear_system.block_size).transpose()

        # Concatenate initial condition (nx, 1) with solution (nx, nz-1)
        initial_condition = probe.reshape(self.linear_system.block_size, 1)
        return np.concatenate([initial_condition, solution], axis=1)

    def _solve_single_probe_iteratively(self, scan_index: int, n: np.ndarray,
                                        reverse: bool = False,
                                        initial_condition: Optional[np.ndarray] = None,
                                        test_impedance: bool = False,
                                        iterative_lu: Tuple[Optional[list[spla.SuperLU]], Optional[list[np.ndarray]], Optional[list[np.ndarray]]] = None
                                        ) -> np.ndarray:
        """Solve the paraxial wave equation iteratively for a single probe."""
        # Initialize solution grid
        solution = np.zeros((self.linear_system.block_size, self.nz),
                            dtype=complex)

        # Set initial condition
        if reverse:
            if initial_condition is None:
                raise ValueError(
                    "Initial condition must be provided for reverse propagation.")
            solution[:, 0] = self._remove_dirichlet_padding(initial_condition[scan_index, ...]).flatten()
        else:
            solution[:, 0] = self.initial_condition.apply_initial_condition(
                scan_index,
                self.thin_sample).flatten()
        
        # Solve with LU decomposition
        if iterative_lu is not None:
            A_lu_list, B_list, b = iterative_lu
            # Iterate over the z dimension
            for j in range(1, self.nz):
                if test_impedance:
                    b = self.linear_system.test_exact_impedance_rhs_slice(j)

                rhs_matrix = B_list.dot(solution[:, j - 1]) + b
                solution[:, j] = A_lu_list[j - 1].solve(rhs_matrix)
        # For thin samples with spsolve
        else:
            object_slices = (
                    self.sample_space.create_sample_slices(
                        self.thin_sample, n=n, scan_index=scan_index
                    )
                ).reshape(-1, self.nz - 1)

            # Create Linear System and Apply Boundary Conditions
            A, B, b = self.linear_system.create_system_slice(
                scan_index=scan_index)

            if reverse:
                A, B = B, A
                b = - b

            # Iterate over the z dimension
            for j in range(1, self.nz):
                if test_impedance:
                    b = self.linear_system.test_exact_impedance_rhs_slice(j)
                
                if reverse:
                    C = - sp.diags(object_slices[:, -j])
                else:
                    C = sp.diags(object_slices[:, j - 1])

                A_with_object = A - C  # LHS Matrix
                B_with_object = B + C  # RHS Matrix
                rhs_matrix = B_with_object.dot(solution[:, j - 1]) + b
                solution[:, j] = spla.spsolve(A_with_object, rhs_matrix)

        return solution

    def return_forward_model_matrix(self, scan_index: int = 0,
                                    n: np.ndarray = None) -> np.ndarray:
        """Return the forward model matrix for a given scan index."""
        # Create the inhomogeneous forward model matrix
        A_homogeneous = (
            self.linear_system.setup_homogeneous_forward_model_lhs(
                scan_index=scan_index)
        )

        Ck = self.linear_system.setup_inhomogeneous_forward_model(
            n=n, scan_index=scan_index)

        return (A_homogeneous - Ck).tocsc()  # Convert to Compressed Sparse Column format for efficiency

    def _create_solution(self) -> np.ndarray:
        """Create the solution grid based on the sample space and boundary 
        conditions."""
        # Initialize solution grid with initial condition
        if self.sample_space.bc_type == "dirichlet":
            u = np.zeros(
                (self.sample_space.num_probes,
                    *self.slice_dimensions_dirichlet, self.nz),
                dtype=complex
            )
        else:
            u = np.zeros(
                (self.sample_space.num_probes,
                    *self.slice_dimensions, self.nz),
                dtype=complex
            )
        return u

    def _handle_dirichlet_bcs(self, solution: np.ndarray) -> np.ndarray:
        """Handle Dirichlet boundary conditions by padding the solution."""
        if self.sample_space.bc_type == "dirichlet":
            shape = (*self.slice_dimensions, self.nz)
            solution = solution.reshape(shape)

            # Build padding with 1-cell Dirichlet padding on spatial dims only
            padding = tuple((1, 1) for _ in self.slice_dimensions) + ((0, 0),)

            assert len(padding) == len(shape), "Padding and array dimensions must match"

            solution = np.pad(
                solution,
                padding,
                mode='constant',
                constant_values=0
            )
        else:
            solution = solution.reshape((*self.slice_dimensions, self.nz))
        return solution

    def _remove_dirichlet_padding(self, solution: np.ndarray) -> np.ndarray:
        """Remove Dirichlet boundary padding from the solution."""
        # Remove 1-cell padding from each spatial dimension
        if self.sample_space.bc_type == "dirichlet":
            if self.sample_space.dimension == 2:
                solution = solution[1:-1, 1:-1]
            elif self.sample_space.dimension == 1:
                solution = solution[1:-1]
        return solution
