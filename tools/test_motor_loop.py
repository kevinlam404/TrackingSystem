from threading import Event, Thread
from time import monotonic

from software.controller import OpenLoopTrackingController, TargetError
from software.gimbal import SimulatedGimbal
from software.motor_loop import MotorLoop
import pytest


def make_loop():
    return MotorLoop(SimulatedGimbal(), OpenLoopTrackingController(), stale_seconds=0.15)


def test_stale_target_holds_without_new_vision_frames():
    loop = make_loop()
    loop._last_tick = 10.0
    loop.submit('track', 10.0, TargetError(0.5, 0.5))
    loop.tick(10.025)
    assert loop.get_status().commanded_pan_speed > 0
    assert loop.get_status().commanded_tilt_degrees > 0
    tilt = loop.get_status().commanded_tilt_degrees
    loop.tick(10.2)
    assert loop.stale
    assert loop.get_status().commanded_pan_speed == 0
    assert loop.get_status().commanded_tilt_degrees == tilt


def test_estop_latched_and_old_results_cannot_restart():
    loop = make_loop()
    loop.submit('scan', 10, scan_speed=0.2)
    loop.tick(10)
    loop.stop_motion(emergency=True)
    loop.resume()
    loop.submit('scan', 10, scan_speed=0.2)
    loop.tick(10.01)
    assert not loop.get_status().enabled
    loop.reset_estop()
    loop.tick(10.02)
    assert not loop.get_status().enabled
    loop.resume()
    loop.submit('scan', 10.03, scan_speed=0.2)
    loop.tick(10.03)
    assert loop.get_status().enabled


def test_normal_stop_blocks_inflight_result():
    loop = make_loop()
    loop.stop_motion()
    loop.submit('track', 10, TargetError(1, 1))
    loop.tick(10)
    assert not loop.get_status().enabled


def test_fault_latched_without_retrying_motion():
    loop = make_loop()
    def fail(speed):
        raise OSError('I2C failed')
    loop.gimbal.set_pan_speed = fail
    loop.submit('scan', 10, scan_speed=0.2)
    loop.tick(10)
    assert loop.fault == 'I2C failed'
    assert not loop.get_status().enabled
    loop.tick(10.01)
    assert not loop.get_status().enabled


def test_background_ticks_when_vision_produces_nothing():
    loop = make_loop()
    loop.submit('scan', monotonic(), scan_speed=0.2)
    loop.start()
    try:
        Event().wait(0.22)
        assert loop.ticks >= 4
        assert loop.get_status().commanded_pan_speed == 0
    finally:
        loop.close()
    assert not loop.gimbal.get_status().enabled


def test_center_holds_tilt_until_explicit_stop():
    loop = make_loop()
    loop.center()
    loop.tick(10)
    loop.tick(10.025)
    assert loop.get_status().enabled
    assert loop.get_status().commanded_tilt_degrees == 0
    loop.stop_motion()
    loop.tick(10.05)
    assert not loop.get_status().enabled


def test_pan_ramps_both_directions_but_stop_is_immediate():
    loop = make_loop()
    loop._last_tick = 10
    loop.submit('scan', 10, scan_speed=1)
    loop.tick(10.025)
    assert loop.get_status().commanded_pan_speed == pytest.approx(0.025)
    loop.submit('scan', 10.025, scan_speed=-1)
    loop.tick(10.05)
    assert loop.get_status().commanded_pan_speed == pytest.approx(0)
    loop.tick(10.075)
    assert loop.get_status().commanded_pan_speed == pytest.approx(-0.025)
    loop.stop_motion()
    loop.tick(10.1)
    assert loop.get_status().commanded_pan_speed == 0


def test_stop_and_diagnostics_do_not_wait_for_blocked_i2c():
    loop = make_loop()
    entered, release, stopped = Event(), Event(), Event()
    original = loop.gimbal.set_pan_speed
    def blocked(speed):
        entered.set()
        assert release.wait(2)
        original(speed)
    loop.gimbal.set_pan_speed = blocked
    loop._last_tick = 10
    loop.submit('track', 10, TargetError(1, 1))
    worker = Thread(target=lambda: loop.tick(10.025))
    worker.start()
    assert entered.wait(1)
    def request_stop():
        loop.stop_motion(emergency=True)
        assert loop.diagnostics()['emergency_stopped']
        stopped.set()
    requester = Thread(target=request_stop)
    requester.start()
    try:
        assert stopped.wait(0.5)
    finally:
        release.set()
        worker.join(2)
        requester.join(2)
    assert not loop.get_status().enabled
    assert loop.get_status().commanded_tilt_degrees == 0


def test_center_cannot_be_overwritten_by_old_vision_result():
    loop = make_loop()
    loop.center()
    loop.submit('idle', 10)
    loop.tick(10)
    loop.submit('idle', 10.025)
    loop.tick(10.025)
    assert loop.get_status().enabled
