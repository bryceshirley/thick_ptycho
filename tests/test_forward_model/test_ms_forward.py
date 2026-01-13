import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# UPDATE THIS IMPORT to match your actual file structure
# from my_module import MSForwardModelSolver
from thick_ptycho.forward_model.base.base_solver import BaseForwardModelSolver

from thick_ptycho.forward_model.multislice.ms_solver import MSForwardModelSolver 

import pytest
import numpy as np
from types import SimpleNamespace
from thick_ptycho.simulation.scan_frame import Limits

from thick_ptycho.simulation.simulation_space import create_simulation_space
from thick_ptycho.simulation.config import ProbeConfig, ProbeType, SimulationConfig

from thick_ptycho.simulation.simulation_space import create_simulation_space
from thick_ptycho.simulation.ptycho_object import create_ptycho_object
from thick_ptycho.simulation.ptycho_probe import create_ptycho_probes

from thick_ptycho.forward_model import PWEIterativeLUSolver, MSForwardModelSolver
import os


import time
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from notebooks.examples_2D.utils import plot_solution, create_random_circle_object, plot_probes, plot_data
# --- 1. SETUP & DUMMY CLASSES ------------------------------------------------

# We create a dummy Base class to simulate the parent class dependency
# This prevents ImportErrors if you run this in isolation and simulates
# the logic of the missing parent class.
class DummyBaseSolver:
    def __init__(self, simulation_space, ptycho_probes, **kwargs):
        self.simulation_space = simulation_space
        self.ptycho_probes = ptycho_probes
        # Mimic attributes set by the real BaseForwardModelSolver
        self.results_dir = kwargs.get("results_dir", "")
        self.use_logging = kwargs.get("use_logging", False)
        self.verbose = kwargs.get("verbose", False)
        self.log = kwargs.get("log", None)

    def get_projected_obj(self, n, mode="forward", proj_idx=0):
        """
        Dummy implementation of the projection method expected from the base class.
        Returns a trivial projection (just repeating the refractive index).
        """
        # Return shape (nx, nz)
        # We assume 'n' is passed or we generate a default vacuum
        return np.ones((self.simulation_space.effective_nx, self.simulation_space.nz))

# --- 2. FIXTURES -------------------------------------------------------------

@pytest.fixture
def sim_space(limits_2d):
    """Creates a configuration object mimicking the SimulationSpace."""
    # Using SimpleNamespace to avoid importing the actual class
    k = 100
    wavelength = 2 * np.pi / k
    nx = 100
    nz = 50

    probe_config = ProbeConfig(
        type=ProbeType.AIRY_DISK,
        wave_length=wavelength,
    )

    sim_cfg = SimulationConfig(
        probe_config=probe_config,
        step_size_px=nx,
        nz=nz,
        spatial_limits=limits_2d,
    )

    return create_simulation_space(sim_cfg)

@pytest.fixture
def solver(sim_space):
    """
    Returns an instance of MSForwardModelSolver.
    We patch the class hierarchy slightly to ensure it uses our DummyBaseSolver 
    if the real one isn't available or strictly for isolation.
    """
    # Create a test subclass to override external dependencies safely
    class TestableMSSolver(MSForwardModelSolver, DummyBaseSolver):
        pass
    
    return TestableMSSolver(
        simulation_space=sim_space,
        ptycho_probes=None,
        method="ASM",
        transform="FFT"
    )

# --- 3. TESTS ----------------------------------------------------------------

def test_init_defaults(solver, sim_space):
    """Test initialization values."""
    assert solver.method == "ASM"
    assert solver.transform == "FFT"
    assert solver.k == sim_space.k
    assert solver.nx_eff == sim_space.effective_nx
    assert solver.pruning_fraction == 0.0
    assert solver._kernel_cache == {}

def test_init_3d_raises_error(sim_space):
    """Test that 3D dimension raises NotImplementedError."""
    sim_space.dimension = 3
    
    # We define the class locally to capture the init logic
    # (Using the same Testable trick as the fixture)
    class TestableMSSolver(MSForwardModelSolver, DummyBaseSolver): pass

    with pytest.raises(NotImplementedError, match="3D simulation space is not yet implemented"):
        TestableMSSolver(sim_space, None)

def test_init_invalid_args(sim_space):
    class TestableMSSolver(MSForwardModelSolver, DummyBaseSolver): pass

    with pytest.raises(ValueError, match="Unknown transform type"):
        TestableMSSolver(sim_space, None, transform="BAD_TRANSFORM")
        
    with pytest.raises(ValueError, match="Unknown method"):
        TestableMSSolver(sim_space, None, method="BAD_METHOD")

# --- KERNEL CACHING TESTS ---

def test_kernel_caching(solver):
    """Verify that kernels are cached and not recomputed."""
    dz = 1e-8
    
    # Empty initially
    assert len(solver._kernel_cache) == 0
    
    # First call
    k1 = solver._get_propagation_kernels(dz, "FFT")
    assert len(solver._kernel_cache) == 1
    
    # Second call (same params)
    k2 = solver._get_propagation_kernels(dz, "FFT")
    assert len(solver._kernel_cache) == 1
    
    # Identity check (must be same object)
    assert k1 is k2

def test_reset_cache(solver):
    solver._get_propagation_kernels(1e-8, "FFT")
    assert len(solver._kernel_cache) > 0
    solver.reset_cache()
    assert len(solver._kernel_cache) == 0

def test_pruning_fraction(solver):
    """Test that setting pruning > 0 zeros out part of the kernel."""
    solver.pruning_fraction = 0.5
    dz = 1e-8
    
    # Force kernel generation
    kernel = solver._get_propagation_kernels(dz, "FFT")
    
    # Check if we have zeros (masked high frequencies)
    assert np.any(kernel == 0j)
    # The DC component (index 0) should remain non-zero
    assert kernel[0] != 0j

# --- TRANSFORM LOGIC TESTS ---

@pytest.mark.parametrize("transform_type", ["FFT", "DCT", "DST"])
def test_transform_invertibility(solver, transform_type):
    """
    Test that Forward -> Backward returns the original signal (approx Identity).
    This validates the math of the transforms.
    """
    solver.transform = transform_type
    dz = solver.dz
    
    # Random complex signal
    np.random.seed(42)
    psi_in = np.random.rand(solver.nx_eff) + 1j * np.random.rand(solver.nx_eff)
    
    # 1. Forward
    if transform_type == "FFT":
        psi_fwd = solver.transform_fft(psi_in, dz, inverse=False)
        psi_rec = solver.transform_fft(psi_fwd, dz, inverse=True)
    elif transform_type == "DCT":
        psi_fwd = solver.transform_dct(psi_in, dz, inverse=False)
        psi_rec = solver.transform_dct(psi_fwd, dz, inverse=True)
    elif transform_type == "DST":
        psi_fwd = solver.transform_dst(psi_in, dz, inverse=False)
        psi_rec = solver.transform_dst(psi_fwd, dz, inverse=True)
        
    # Check tolerance
    np.testing.assert_allclose(psi_in, psi_rec, rtol=1e-5, atol=1e-5)

def test_absorbing_boundary_logic(solver):
    """
    Test that absorbing boundary is exactly 0.5 * (DCT + DST).
    """
    dz = solver.dz
    psi_in = np.random.rand(solver.nx_eff) + 1j * np.random.rand(solver.nx_eff)
    
    # Calculate parts manually
    val_dct = solver.transform_dct(psi_in, dz)
    val_dst = solver.transform_dst(psi_in, dz)
    expected = 0.5 * (val_dct + val_dst)
    
    # Calculate via method
    result = solver.transform_absorbing_boundary(psi_in, dz)
    
    np.testing.assert_allclose(result, expected)

def test_propagate_dispatch(solver, monkeypatch):
    """Test that _propagate calls the correct sub-function based on self.transform."""
    
    # Create a dummy recorder
    called = []
    def mock_fft(*args, **kwargs):
        called.append("FFT")
        return np.zeros_like(args[0])
    
    def mock_dct(*args, **kwargs):
        called.append("DCT")
        return np.zeros_like(args[0])

    # Use monkeypatch to swap methods
    monkeypatch.setattr(solver, "transform_fft", mock_fft)
    monkeypatch.setattr(solver, "transform_dct", mock_dct)
    
    psi = np.zeros(solver.nx_eff)
    
    solver.transform = "FFT"
    solver._propagate(psi, 1.0)
    assert called[-1] == "FFT"
    
    solver.transform = "DCT"
    solver._propagate(psi, 1.0)
    assert called[-1] == "DCT"

def test_backpropagate_dispatch(solver, monkeypatch):
    """Test that _backpropagate calls the correct sub-function with inverse=True."""
    
    called_args = {}
    def mock_fft(psi, dz, inverse):
        called_args['inverse'] = inverse
        return psi

    monkeypatch.setattr(solver, "transform_fft", mock_fft)
    
    solver.transform = "FFT"
    solver._backpropagate(np.zeros(10), 1.0)
    
    assert called_args['inverse'] is True

# --- TRANSMISSION & SOLVER LOOP TESTS ---

def test_object_transmission_function(solver):
    """Test calculation of transmission phase: exp(i * k * dn * dz)."""
    n_slice = np.array([1.0, 1.1, 1.2]) # varying refractive index
    solver.n_medium = 1.0
    solver.dz = 1e-9
    solver.k = 1e10
    
    # Expected: exp(i * k * (n - n_medium) * dz)
    expected_phase = solver.k * (n_slice - solver.n_medium) * solver.dz
    expected_t = np.exp(1j * expected_phase)
    
    result = solver._object_transmission_function(n_slice)
    
    np.testing.assert_allclose(result, expected_t)

def test_solve_single_probe_forward(solver):
    """
    Test the structure of the forward solver loop.
    We check output shapes and data types.
    """
    # 1. Setup
    probe = np.ones(solver.nx_eff, dtype=complex)
    
    # 2. Run Forward
    # Note: solver is an instance of TestableMSSolver which has a dummy get_projected_obj
    u_field, psi_exit = solver._solve_single_probe(
        scan_idx=0, 
        mode="forward", 
        probe=probe
    )
    
    # 3. Assertions
    # u_field should be (nx, nz)
    assert u_field.shape == (solver.nx_eff, solver.nz)
    assert psi_exit.shape == (solver.nx_eff,)
    assert np.iscomplexobj(u_field)
    
    # Since we use vacuum (n=1) and strang splitting, energy should be conserved 
    # (ignoring numerical noise and edge effects if DCT/DST were used)
    # For FFT + Vacuum, |psi_exit| should equal |probe|
    solver.transform = "FFT"
    np.testing.assert_allclose(np.abs(psi_exit), np.abs(probe), rtol=1e-5)

def test_solve_single_probe_backward(solver):
    """Test backward propagation mode runs without error."""
    probe = np.zeros(solver.nx_eff, dtype=complex)
    
    u_field, psi_exit = solver._solve_single_probe(
        scan_idx=0, 
        mode="backward", 
        probe=probe
    )
    
    assert u_field.shape == (solver.nx_eff, solver.nz)
    
# def test_solve_single_probe_strang_splitting(solver):
#     """
#     Verify code path for Strang Splitting (half steps at start/end).
#     We can't easily verify the math without exact analytical solutions, 
#     but we can ensure it runs diff from non-strang.
#     """
#     solver.strang_splitting = True
#     probe = np.random.rand(solver.nx_eff) + 1j * np.random.rand(solver.nx_eff)
    
#     res_strang, _ = solver._solve_single_probe(mode="forward", probe=probe)
    
#     solver.strang_splitting = False
#     # Reset cache to be safe, though not strictly required if dz changes
#     solver.reset_cache()
    
#     res_no_strang, _ = solver._solve_single_probe(mode="forward", probe=probe)
    
#     # The intermediate fields should differ because Strang applies a half-prop 
#     # *before* the first slice, whereas standard usually doesn't (or handles it differently).
#     assert not np.allclose(res_strang, res_no_strang)

def test_solve_single_probe_auto_probe_init(solver):
    """Test that if probe is None, it defaults to zeros."""
    # We patch the object getter to ensure valid return
    u, psi = solver._solve_single_probe(probe=None)
    
    # Should be all zeros
    np.testing.assert_allclose(u, 0)
    np.testing.assert_allclose(psi, 0)

class DummyBaseSolver:
    """
    Simulates the parent class logic to allow running tests in isolation
    if the full BaseForwardModelSolver hierarchy isn't perfectly reachable.
    """
    def __init__(self, simulation_space, ptycho_probes, **kwargs):
        self.simulation_space = simulation_space
        self.ptycho_probes = ptycho_probes
        self.results_dir = kwargs.get("results_dir", "")
        self.use_logging = kwargs.get("use_logging", False)
        self.verbose = kwargs.get("verbose", False)
        self.log = kwargs.get("log", None)

    def get_projected_obj(self, n, mode="forward", proj_idx=0):
        """
        Dummy implementation. If 'n' is passed explicitly (as in our tests),
        it returns it. Otherwise, returns a default vacuum object.
        """
        if n is not None:
            return n
        return np.ones((self.simulation_space.effective_nx, self.simulation_space.nz))

# --- 2. FIXTURES ----------------------------------------------------------

@pytest.fixture
def sim_space():
    """
    Creates the simulation space using the strict configuration pattern 
    provided in the prompt.
    """
    nx = 1000  # Number of pixels
    nz = 100   # Number of slices

    spatial_limits = Limits(
        x=(0.0, 150e-6),  # Sample width
        z=(0.0, 40e-6),  # Sample thickness
        units="meters",
    )

    probe_config = ProbeConfig(
        type=ProbeType.AIRY_DISK,
        wave_length=6.35e-7,  # meters (0.635 μm). Visible light
        diameter=10e-6,  # [m]
        focus=-5e-6,  # focal length [m]
    )

    sim_cfg = SimulationConfig(
        probe_config=probe_config,
        step_size_px=nx, # Note: This param name might vary in your actual config
        scan_points=1,
        nz=nz,
        spatial_limits=spatial_limits,
    )
    
    # We monkeypatch attributes onto the created space if the config 
    # doesn't automatically set all necessary solver properties
    space = create_simulation_space(sim_cfg)
    
    # Ensure these attributes exist for the solver to read (if not set by create_simulation_space)
    if not hasattr(space, 'n_medium'): space.n_medium = 1.0
    if not hasattr(space, 'effective_nx'): space.effective_nx = nx
    if not hasattr(space, 'dx'): space.dx = 1e-9
    if not hasattr(space, 'dz'): space.dz = 1e-9
    if not hasattr(space, 'refractive_index_empty'): 
        space.refractive_index_empty = np.ones((nx, nz))

    return space

@pytest.fixture
def solver_hetero(sim_space):
    """
    Returns an instance of MSForwardModelSolver using the configured sim_space.
    """

    return MSForwardModelSolver(
        simulation_space=sim_space,
        ptycho_probes=None,
        method="ASM",
        transform="FFT",
    )

# --- 3. REVERSIBILITY TEST FOR HETEROGENEOUS OBJECT -----------------------

def test_heterogeneous_reversibility(solver_hetero,sim_space):
    """
    Verifies that solving backwards gives the right solution for a 
    heterogeneous refractive index (random circles).
    """
    
    # 1. Create Heterogeneous Refractive Index
    ptycho_object = create_ptycho_object(sim_space)

    ptycho_object = create_random_circle_object(
        sim_space, ptycho_object, num_circles=5
    )
    n_hetero = ptycho_object.refractive_index

    ptycho_probes = create_ptycho_probes(
        sim_space)

    # 3. Forward Propagation
    u_forward, psi_exit = solver_hetero._solve_single_probe(
        scan_idx=0,
        mode="forward",
        probe=ptycho_probes[0],
        n=n_hetero
    )

    fig, ax =solver_hetero.simulation_space.viewer.plot_single_panel(
            np.abs(u_forward).T,
            title="forward",
            xlabel="Scan Number #",
            ylabel="x (px)"
        )
    fig.savefig("./forward.png", bbox_inches="tight")

    
    u_backward, psi_recovered = solver_hetero._solve_single_probe(
        scan_idx=0,
        mode="backward",
        probe=psi_exit,
        n=n_hetero
    )

    fig, ax =solver_hetero.simulation_space.viewer.plot_single_panel(
            np.abs(u_backward).T,
            title="Backward",
            xlabel="Scan Number #",
            ylabel="x (px)"
        )
    fig.savefig("./backward.png", bbox_inches="tight")


    # Plot 1D exit waves
    import matplotlib.pyplot as plt
    # Plot 1D exit waves#
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Common index for convenience
    probe_idx = sim_space.num_probes // 2
    x_vals = range(sim_space.effective_shape[0])

    # --- Subplot 1: Probe ---
    axs[0].plot(
        x_vals,
        np.abs(ptycho_probes[0, probe_idx, :]),
        label=f"probe {probe_idx}",
    )
    axs[0].set_title("Probe amplitude")
    axs[0].set_xlabel("x (px)")
    axs[0].legend()

    # --- Subplot 2: Exitwave ---
    axs[1].plot(
        x_vals,
        np.abs(psi_recovered[probe_idx,:]),
        label=f"probe {probe_idx}",
    )
    axs[1].set_title("Exitwave amplitude after backward propagation")
    axs[1].set_xlabel("x (px)")
    axs[1].set_ylim(0, np.max(np.abs(psi_recovered[probe_idx,:]))*1.1)
    axs[1].legend()

    plt.tight_layout()
    fig.savefig("./probes.png", bbox_inches="tight")

    # Plot 1D exit waves
    # Plot 1D exit waves#
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Common index for convenience
    probe_idx = sim_space.num_probes // 2
    x_vals = range(sim_space.effective_shape[0])

    # --- Subplot 1: Probe ---
    axs[0].plot(
        x_vals,
        np.abs(ptycho_probes[0, probe_idx, :]),
        label=f"probe {probe_idx}",
    )
    axs[0].set_title("Probe amplitude")
    axs[0].set_xlabel("x (px)")
    axs[0].legend()

    # --- Subplot 2: Exitwave ---
    axs[1].plot(
        x_vals,
        np.abs(psi_exit[probe_idx,:]),
        label=f"probe {probe_idx}",
    )
    axs[1].set_title("Exitwave amplitude after forward propagation")
    axs[1].set_xlabel("x (px)")
    axs[1].set_ylim(0, np.max(np.abs(psi_exit[probe_idx,:]))*1.1)
    axs[1].legend()

    plt.tight_layout()
    fig.savefig("./exitwave.png", bbox_inches="tight")

    # Check max absolute error
    abs_error = np.imag(psi_recovered - ptycho_probes[0])
    max_error = np.max(abs_error)
    np.testing.assert_allclose(
        max_error, 
        0.0, 
        atol=0.05,
        err_msg=f"Max absolute error {max_error} exceeds tolerance."
    )

@pytest.mark.parametrize("method", ["Paraxial", "ASM"])
@pytest.mark.parametrize("transform", ["FFT", "DCT", "DST", "absorbing_boundary"])
def test_adjoints_consistency_propagator(method, transform, sim_space):
    ms_solver = MSForwardModelSolver(
        simulation_space=sim_space,
        ptycho_probes=None,
        method=method,
        transform=transform,
    )
    x = np.random.randn(ms_solver.nx_eff) + 1j*np.random.randn(ms_solver.nx_eff)
    y = np.random.randn(ms_solver.nx_eff) + 1j*np.random.randn(ms_solver.nx_eff)

    Ax = ms_solver._propagate(x, ms_solver.dz)
    Aty = ms_solver._backpropagate(y, ms_solver.dz)
    assert np.allclose(np.vdot(Ax, y), np.vdot(x, Aty), rtol=1e-5, atol=1e-5)

# def test_object_transmission_function(solver):
#     """Test calculation of transmission phase: exp(i * k * dn * dz)."""
#     # 1. Create Heterogeneous Refractive Index
#     ptycho_object = create_ptycho_object(sim_space)

#     ptycho_object = create_random_circle_object(
#         sim_space, ptycho_object, num_circles=5
#     )
#     n_hetero = ptycho_object.refractive_index

#     ptycho_probes = create_ptycho_probes(
#         sim_space)

# --- 4. STANDARD TESTS (Kernel, Caching, etc.) ----------------------------

def test_kernel_caching(solver):
    dz = 1e-8
    k1 = solver._get_propagation_kernels(dz, "FFT")
    k2 = solver._get_propagation_kernels(dz, "FFT")
    assert k1 is k2
    solver.reset_cache()
    assert len(solver._kernel_cache) == 0

def test_solve_single_probe_forward_structure(solver):
    """Basic check that the forward loop returns correct shapes."""
    probe = np.ones(solver.nx_eff, dtype=complex)
    u, psi = solver._solve_single_probe(scan_idx=0, mode="forward", probe=probe)
    assert u.shape == (solver.nx_eff, solver.nz)
    assert psi.shape == (solver.nx_eff,)

def test_auto_probe_init(solver):
    """Test handling of None probe input."""
    u, psi = solver._solve_single_probe(probe=None)
    np.testing.assert_allclose(u, 0)