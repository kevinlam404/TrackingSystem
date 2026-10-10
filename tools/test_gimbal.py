import pytest
from software.gimbal import PCA9685Gimbal, SimulatedGimbal

def test_gimbal_starts_disabled() -> None:
    gimbal = SimulatedGimbal()
    status = gimbal.get_status()

    assert status.enabled is False
    assert status.commanded_pan_speed == 0.0
    assert status.commanded_tilt_degrees == 0.0

def test_disabled_gimbal_rejects_movement() -> None:
    gimbal = SimulatedGimbal()
    with pytest.raises(RuntimeError):
        gimbal.set_pan_speed(0.2)
    with pytest.raises(RuntimeError):
        gimbal.set_tilt_angle(20.0)

def test_pan_speed_is_limited() -> None:
    gimbal = SimulatedGimbal()
    gimbal.enable()
    gimbal.set_pan_speed(2.0)

    assert(gimbal.get_status().commanded_pan_speed == 1.0)
    gimbal.set_pan_speed(-2.0)
    assert(gimbal.get_status().commanded_pan_speed == -1.0)

def test_tilt_angle_is_limited() -> None:
    gimbal = SimulatedGimbal(tilt_min_degrees=-80.0, tilt_max_degrees=80.0)
    gimbal.enable()
    gimbal.set_tilt_angle(100.0)
    assert(gimbal.get_status().commanded_tilt_degrees == 80.0)
    gimbal.set_tilt_angle(-100.0)
    assert(gimbal.get_status().commanded_tilt_degrees == -80.0)

def test_hold_stops_pan_and_keeps_tilt() -> None:
    gimbal = SimulatedGimbal()
    gimbal.enable()

    gimbal.set_pan_speed(0.4)
    gimbal.set_tilt_angle(25.0)
    gimbal.hold()
    status = gimbal.get_status()
    assert status.commanded_pan_speed == 0.0
    assert status.commanded_tilt_degrees == 25.0

def test_center_stops_pan_and_centers_tilt() -> None:
    gimbal = SimulatedGimbal()
    gimbal.enable()

    gimbal.set_pan_speed(-0.4)
    gimbal.set_tilt_angle(30.0)

    gimbal.center()
    status = gimbal.get_status()
    
    assert status.commanded_pan_speed == 0.0
    assert status.commanded_tilt_degrees == 0.0

def test_disable_stops_pan() -> None:
    gimbal = SimulatedGimbal()
    gimbal.enable()

    gimbal.set_pan_speed(0.5)
    gimbal.disable()

    status = gimbal.get_status()

    assert status.enabled is False
    assert status.commanded_pan_speed == 0.0


def test_pan_stop_pulse_maps_to_neutral_throttle() -> None:
    assert PCA9685Gimbal._pulse_to_throttle(1500, 1000, 2000) == 0.0


def test_offset_pan_stop_pulse_maps_to_offset_throttle() -> None:
    assert PCA9685Gimbal._pulse_to_throttle(1510, 1000, 2000) == pytest.approx(0.02)


def test_pan_pulse_mapping_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError):
        PCA9685Gimbal._pulse_to_throttle(2200, 1000, 2000)
