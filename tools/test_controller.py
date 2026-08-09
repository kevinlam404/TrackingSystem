import pytest
from software.controller import OpenLoopTrackingController,TargetError,calculate_target_error
from software.detector import Detection

def make_target(center_x: int, center_y: int, width: int = 40, height: int = 40) -> Detection:
    return Detection(
        x1=center_x - width // 2,
        y1=center_y - height // 2,
        x2=center_x + width // 2,
        y2=center_y + height // 2,
        confidence = 1.0,
        class_id=0,
        label="person",
        track_id = 1
    )

def test_centered_target_has_zero_error() -> None:
    target = make_target(center_x = 320, center_y = 240)
    error = calculate_target_error(target=target, frame_width=640, frame_height=480)
    assert error.pan == 0.0
    assert error.tilt == 0.0

def test_target_on_right_has_positive_pan() -> None:
    target = make_target(center_x=500, center_y=240)
    error = calculate_target_error(target=target,frame_width=640,frame_height=480)
    assert error.pan > 0
    assert error.horizontal_direction == "RIGHT"

def test_target_on_left_has_negative_pan() -> None:
    target = make_target(center_x=100, center_y=240)
    error = calculate_target_error(target=target,frame_width=640,frame_height=480)
    assert error.pan < 0
    assert error.horizontal_direction == "LEFT"

def test_target_above_has_positive_tilt() -> None:
    target = make_target(center_x=320, center_y=100)
    error = calculate_target_error(target=target,frame_width=640,frame_height=480)
    assert error.tilt > 0
    assert error.vertical_direction == "UP"

def test_target_below_has_negative_tilt() -> None:
    target = make_target(center_x=320, center_y=400)
    error = calculate_target_error(target=target,frame_width=640,frame_height=480)
    assert error.tilt < 0
    assert error.vertical_direction == "DOWN"

def test_deadzone_removes_small_error() -> None:
    target = make_target(center_x=325, center_y=235)
    error = calculate_target_error(target=target,frame_width=640,frame_height=480, dead_zone=0.05)
    assert error.pan == 0.0
    assert error.tilt == 0.0

def test_controller_applies_minimum_pan_speed() -> None:
    controller = OpenLoopTrackingController(maximum_pan_speed=0.4, minimum_pan_speed=0.08)
    command = controller.update(target_error=TargetError(pan=0.1,tilt=0.0),current_tilt_degrees=0.0,elapsed_seconds=0.1)
    assert command.pan_speed == pytest.approx(0.08)

def test_controller_limits_maximum_pan_speed() -> None:
    controller = OpenLoopTrackingController(maximum_pan_speed=0.4,minimum_pan_speed=0.08)
    command = controller.update(target_error=TargetError(pan=2.0,tilt=0.0),current_tilt_degrees=0.0,elapsed_seconds=0.1)
    assert command.pan_speed == pytest.approx(0.4)

def test_controller_preserves_pan_direction() -> None:
    controller = OpenLoopTrackingController()
    command = controller.update(target_error=TargetError(pan=-0.5,tilt=0.0), current_tilt_degrees=0.0,elapsed_seconds=0.1)
    assert command.pan_speed < 0

def test_tilt_changes_using_elapsed_time() -> None:
    controller = OpenLoopTrackingController(tilt_speed_dps=45.0)
    command = controller.update(target_error=TargetError(pan=0.0,tilt=0.5),current_tilt_degrees=0.0,elapsed_seconds=0.1)
    assert command.tilt_degrees == pytest.approx(2.25)

def test_tilt_is_limited() -> None:
    controller = OpenLoopTrackingController(tilt_speed_dps=45.0,tilt_min_degrees=-80.0,tilt_max_degrees=80.0)
    upward = controller.update(target_error=TargetError(pan=0.0,tilt=1.0),current_tilt_degrees=79.0,elapsed_seconds=0.1)
    downward = controller.update(target_error=TargetError(pan=0.0,tilt=-1.0),current_tilt_degrees=-79.0,elapsed_seconds=0.1)
    assert upward.tilt_degrees == 80.0
    assert downward.tilt_degrees == -80.0


    

