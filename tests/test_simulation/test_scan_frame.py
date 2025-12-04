import pytest

from thick_ptycho.simulation.scan_frame import Limits, Point, ScanFrame


def test_pixel_limits():
    limits = Limits(x=(10, 50), units="pixels")
    assert limits.x == (10, 50)
    assert limits.units == "pixels"
    with pytest.raises(ValueError):
        Limits(x=(50, 10), units="pixels")  # Invalid: decreasing limits
    with pytest.raises(ValueError):
        Limits(x=(-5, 10), units="pixels")  # Invalid: negative limits
    with pytest.raises(ValueError):
        Limits(x=(10.5, 50.0), units="pixels")  # Invalid: non-integer limits
    with pytest.raises(ValueError):
        Limits(x=(10, "a"), units="pixels")  # Invalid: non-integer limits
    with pytest.raises(ValueError):
        Limits(x=(10, 50), units="invalid_unit")  # Invalid: unknown units

def test_meter_limits():
    limits = Limits(x=(0.0, 0.1), units="meters")
    assert limits.x == (0.0, 0.1)
    assert limits.units == "meters"
    with pytest.raises(ValueError):
        Limits(x=(0.1, 0.0), units="meters")  # Invalid: decreasing limits
    with pytest.raises(ValueError):
        Limits(x=("a", 0.1), units="meters")  # Invalid: non-numeric limits
    with pytest.raises(ValueError):
        Limits(x=(0.0, None), units="meters")  # Invalid: non-numeric limits

def test_as_tuple():
    limits1 = Limits(x=(0, 10), units="pixels")
    assert limits1.as_tuple() == ((0, 10),)
    
    limits2 = Limits(x=(0.0, 0.1), units="meters")
    assert limits2.as_tuple() == ((0.0, 0.1),)

    limits3 = Limits(x=(0, 10), y=(0,10), z=(0,10), units="pixels")
    assert limits3.as_tuple() == ((0, 10), (0,10), (0,10))

    limit4 = Limits(x=(0.0, 0.1), z=(0.0,0.1), units="meters")
    assert limit4.as_tuple() == ((0.0, 0.1), (0.0,0.1))

def test_point_as_tuple():
    point1 = Point(x=5)
    assert point1.as_tuple() == (5,)

    point2 = Point(x=3.5, y=7.2)
    assert point2.as_tuple() == (3.5, 7.2)

def test_scan_frame():
    scan_frame = ScanFrame(
        probe_centre_continuous=Point(x=0.05),
        probe_centre_discrete=Point(x=500)
    )
    assert scan_frame.probe_centre_continuous.x == 0.05
    assert scan_frame.probe_centre_discrete.x == 500

    scan_frame.set_reduced_limits_continuous(Limits(x=(0.04, 0.06), units="meters"))
    scan_frame.set_reduced_limits_discrete(Limits(x=(400, 600), units="pixels"))

    assert scan_frame.reduced_limits_continuous.x == (0.04, 0.06)
    assert scan_frame.reduced_limits_discrete.x == (400, 600)

    with pytest.raises(ValueError):
        scan_frame.set_reduced_limits_continuous(Limits(x=(0.06, 0.10), units="pixels"))  # Invalid units
    with pytest.raises(ValueError):
        scan_frame.set_reduced_limits_discrete(Limits(x=(400, 600), units="meters"))  # Invalid limits

    

