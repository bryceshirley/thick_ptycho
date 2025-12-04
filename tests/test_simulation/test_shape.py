# test_optical_shape.py
import numpy as np
import pytest

from thick_ptycho.simulation.ptycho_object import (OpticalShape,
                                                   OpticalShape2D,
                                                   OpticalShape3D)


# ------------------------------------------------------------------------------
# 1. Test instantiation behaviour
# ------------------------------------------------------------------------------
def test_opticalshape_creates_1d_instance(dummy_sim_space_2d):
    shape = OpticalShape(
        centre_scale=[0.5, 0.5],
        shape="circle",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=dummy_sim_space_2d,
    )
    assert isinstance(shape, OpticalShape2D), "Should create OpticalShape2D for 1D centre input."


def test_opticalshape_creates_2d_instance(dummy_sim_space_3d):
    shape = OpticalShape(
        centre_scale=[0.5, 0.5, 0.5],
        shape="cuboid",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=dummy_sim_space_3d,
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
def test_invalid_scale_values_raise_assertion(dummy_sim_space_2d, centre_scale, side_length_scale, depth_scale):
    with pytest.raises(AssertionError):
        OpticalShape(
            centre_scale=centre_scale,
            shape="rectangle",
            refractive_index=1.5,
            side_length_scale=side_length_scale,
            depth_scale=depth_scale,
            guassian_blur=None,
            simulation_space=dummy_sim_space_2d,
        )


# ------------------------------------------------------------------------------
# 3. Test out-of-bounds object placement
# ------------------------------------------------------------------------------

def test_object_out_of_bounds_raises_assertion(dummy_sim_space_2d):
    # place centre in the middle but make it too large
    with pytest.raises(AssertionError):
        OpticalShape(
            centre_scale=[0.5, 0.5],
            shape="rectangle",
            refractive_index=1.5,
            side_length_scale=0.9,
            depth_scale=0.9,
            guassian_blur=None,
            simulation_space=dummy_sim_space_2d,
        )


def test_object_out_of_bounds_raises_assertion_2d(dummy_sim_space_3d):
    # place centre at 0.5, 0.5, 0.5 but side/depth too large
    with pytest.raises(AssertionError):
        OpticalShape(
            centre_scale=[0.5, 0.5, 0.5],
            shape="cuboid",
            refractive_index=1.5,
            side_length_scale=0.9,
            depth_scale=0.9,
            guassian_blur=None,
            simulation_space=dummy_sim_space_3d,
        )


# ------------------------------------------------------------------------------
# 4. Tests for refractive index field generation
# ------------------------------------------------------------------------------

def test_field_shape_matches_simulation_space_2d(dummy_sim_space_2d):
    shape = OpticalShape(
        centre_scale=[0.5, 0.5],
        shape="circle",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=dummy_sim_space_2d,
    )
    n_field = shape.get_refractive_index_field()
    assert n_field.shape == (64, 64)
    assert np.iscomplexobj(n_field)


def test_field_shape_matches_simulation_space_3d(dummy_sim_space_3d):
    shape = OpticalShape(
        centre_scale=[0.5, 0.5, 0.5],
        shape="cuboid",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=dummy_sim_space_3d,
    )
    n_field = shape.get_refractive_index_field()
    assert n_field.shape == (64, 64, 64)
    assert np.iscomplexobj(n_field)

# ------------------------------------------------------------------------------
# 5. Tests for Gaussian blur application
# ------------------------------------------------------------------------------

def test_gaussian_blur_applied_changes_field(dummy_sim_space_2d):
    shape_blur = OpticalShape(
        centre_scale=[0.5, 0.5],
        shape="circle",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=2,
        simulation_space=dummy_sim_space_2d,
    )
    shape_no_blur = OpticalShape(
        centre_scale=[0.5, 0.5],
        shape="circle",
        refractive_index=1.5,
        side_length_scale=0.2,
        depth_scale=0.3,
        guassian_blur=None,
        simulation_space=dummy_sim_space_2d,
    )

    field_blur = shape_blur.get_refractive_index_field()
    field_no_blur = shape_no_blur.get_refractive_index_field()

    assert not np.allclose(field_blur, field_no_blur), "Gaussian blur should alter field distribution."


def test_invalid_shape_raises_valueerror(dummy_sim_space_2d):
    with pytest.raises(ValueError):
        OpticalShape(
            centre_scale=[0.5, 0.5],
            shape="unknown_shape",
            refractive_index=1.5,
            side_length_scale=0.3,
            depth_scale=0.3,
            guassian_blur=None,
            simulation_space=dummy_sim_space_2d,
        ).get_refractive_index_field()
