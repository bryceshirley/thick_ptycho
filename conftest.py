import pytest
import numpy as np
from thick_ptycho.simulation.scan_frame import Limits

def pytest_addoption(parser):
    """Add a command-line option to control plotting."""
    parser.addoption(
        "--plot", action="store_true", default=False, help="Enable plotting of convergence plots"
    )
    parser.addoption(
        "--plot_error", action="store_true", default=False, help="Enable plotting of error"
    )
    parser.addoption(
        "--plot_probe_and_ew", action="store_true", default=False, help="Enable plotting of probe and exit wave amplitudes"
    )

class DummySimulationSpace2D:
    """Dummy 1D simulation space for testing OpticalShape2D."""
    def __init__(self, nx=64, nz=64, spatial_limits=Limits(x=(0, 1),z=(0, 1))):
        self.nx = nx
        self.nz = nz
        self.x = np.linspace(*spatial_limits.x, nx)
        self.z = np.linspace(*spatial_limits.z, nz)
        self.dimension = 2

        self.spatial_limits = spatial_limits
        self.shape = (nx, nz)


class DummySimulationSpace3D:
    """Dummy 2D simulation space for testing OpticalShape3D."""
    def __init__(self, nx=64, ny=64, nz=64,
                 spatial_limits=Limits(x=(0, 1),
                                       y=(0, 1),
                                       z=(0, 1))):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.x = np.linspace(*spatial_limits.x, nx)
        self.y = np.linspace(*spatial_limits.y, ny)
        self.z = np.linspace(*spatial_limits.z, nz)
        self.dimension = 3
        self.spatial_limits = spatial_limits
        self.shape = (nx, ny, nz)

@pytest.fixture(scope="session")
def dummy_sim_space_2d():
    sim = DummySimulationSpace2D()
    return sim

@pytest.fixture(scope="session")
def dummy_sim_space_3d():
    sim = DummySimulationSpace3D()
    return sim