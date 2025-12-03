# test_optical_shape.py
import numpy as np
import pytest
from thick_ptycho.simulation.ptycho_object import OpticalShape, OpticalShape2D, OpticalShape3D
from thick_ptycho.simulation.scan_frame import Limits
# Create a dummy simulation 1D space class for testing it should have all the parameters referenced in the OpticalShape class
import numpy as np

class SimulationSpace2D:
    """Dummy 1D simulation space for testing OpticalShape2D."""
    def __init__(self, nx=100, nz=100, spatial_limits=Limits(x=(0, 1),z=(0, 1))):
        self.nx = nx
        self.nz = nz
        self.x = np.linspace(*spatial_limits.x, nx)
        self.z = np.linspace(*spatial_limits.z, nz)
        self.dimension = 2

        self.spatial_limits = spatial_limits
        self.shape = (nx, nz)


class SimulationSpace3D:
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


# ------------------------------------------------------------------------------
# 1. Test instantiation behaviour
# ------------------------------------------------------------------------------

def test_opticalshape_creates_1d_instance():
    sim = SimulationSpace2D()
    shape = OpticalShape(
        centre_scale=[0.5, 0.5],
        shape="circle",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=sim,
    )
    assert isinstance(shape, OpticalShape2D), "Should create OpticalShape2D for 1D centre input."


def test_opticalshape_creates_2d_instance():
    sim = SimulationSpace3D()
    shape = OpticalShape(
        centre_scale=[0.5, 0.5, 0.5],
        shape="cuboid",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=sim,
    )
    assert isinstance(shape, OpticalShape3D), "Should create OpticalShape3D for 2D centre input."


# ------------------------------------------------------------------------------
# 2. Test validation of input scales
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "centre_scale, side_length_scale, depth_scale",
    [
        ([-0.1, 0.5], 0.5, 0.5),
        ([1.2, 0.5], 0.5, 0.5),
        ([0.5, 0.5], -0.2, 0.5),
        ([0.5, 0.5], 0.5, 1.5),
    ],
)
def test_invalid_scale_values_raise_assertion(centre_scale, side_length_scale, depth_scale):
    sim = SimulationSpace2D()
    with pytest.raises(AssertionError):
        OpticalShape(
            centre_scale=centre_scale,
            shape="rectangle",
            refractive_index=1.5,
            side_length_scale=side_length_scale,
            depth_scale=depth_scale,
            guassian_blur=None,
            simulation_space=sim,
        )


# ------------------------------------------------------------------------------
# 3. Test out-of-bounds object placement
# ------------------------------------------------------------------------------

def test_object_out_of_bounds_raises_assertion():
    sim = SimulationSpace2D(nx=100, nz=100)
    # place centre in the middle but make it too large
    with pytest.raises(AssertionError):
        OpticalShape(
            centre_scale=[0.5, 0.5],
            shape="rectangle",
            refractive_index=1.5,
            side_length_scale=0.9,
            depth_scale=0.9,
            guassian_blur=None,
            simulation_space=sim,
        )


def test_object_out_of_bounds_raises_assertion_2d():
    sim = SimulationSpace3D(nx=50, ny=50, nz=50)
    # place centre at 0.5, 0.5, 0.5 but side/depth too large
    with pytest.raises(AssertionError):
        OpticalShape(
            centre_scale=[0.5, 0.5, 0.5],
            shape="cuboid",
            refractive_index=1.5,
            side_length_scale=0.9,
            depth_scale=0.9,
            guassian_blur=None,
            simulation_space=sim,
        )


# ------------------------------------------------------------------------------
# 4. Tests for refractive index field generation
# ------------------------------------------------------------------------------

def test_field_shape_matches_simulation_space_1d():
    sim = SimulationSpace2D(nx=64, nz=64)
    shape = OpticalShape(
        centre_scale=[0.5, 0.5],
        shape="circle",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=sim,
    )
    n_field = shape.get_refractive_index_field()
    assert n_field.shape == (64, 64)
    assert np.iscomplexobj(n_field)


def test_field_shape_matches_simulation_space_2d():
    sim = SimulationSpace3D(nx=32, ny=32, nz=32)
    shape = OpticalShape(
        centre_scale=[0.5, 0.5, 0.5],
        shape="cuboid",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=sim,
    )
    n_field = shape.get_refractive_index_field()
    assert n_field.shape == (32, 32, 32)
    assert np.iscomplexobj(n_field)

# ------------------------------------------------------------------------------
# 5. Tests for Gaussian blur application
# ------------------------------------------------------------------------------

def test_gaussian_blur_applied_changes_field():
    sim = SimulationSpace2D(nx=64, nz=64)
    shape_blur = OpticalShape(
        centre_scale=[0.5, 0.5],
        shape="circle",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=2,
        simulation_space=sim,
    )
    shape_no_blur = OpticalShape(
        centre_scale=[0.5, 0.5],
        shape="circle",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=sim,
    )

    field_blur = shape_blur.get_refractive_index_field()
    field_no_blur = shape_no_blur.get_refractive_index_field()

    assert not np.allclose(field_blur, field_no_blur), "Gaussian blur should alter field distribution."


def test_invalid_shape_raises_valueerror():
    sim = SimulationSpace2D()
    with pytest.raises(ValueError):
        OpticalShape(
            centre_scale=[0.5, 0.5],
            shape="unknown_shape",
            refractive_index=1.5,
            side_length_scale=0.3,
            depth_scale=0.3,
            guassian_blur=None,
            simulation_space=sim,
        ).get_refractive_index_field()
