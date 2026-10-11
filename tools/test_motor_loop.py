from threading import Event, Thread
from time import monotonic

from software.controller import OpenLoopTrackingController, TargetError
from software.gimbal import SimulatedGimbal
from software.motor_loop import MotorLoop
import pytest


def make_loop():
    return MotorLoop(SimulatedGimbal(), OpenLoopTrackingController(), stale_seconds=0.15)


def test_old_resume_generation_cannot_clear_new_stop():
    loop = make_loop()
    generation = loop.generation()
    loop.stop_motion()
    assert not loop.resume(generation)
    loop.submit('scan', 10, scan_speed=.2)
    loop.tick(10)
    assert not loop.get_status().enabled
    assert loop.resume(loop.generation())
    loop.submit('scan', 10.025, scan_speed=.2)
    loop.tick(10.025)
    assert loop.get_status().enabled


def test_latency_measured_once_per_observation_and_not_for_stale(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr('software.motor_loop.monotonic', lambda: clock[0])
    loop = make_loop()
    loop.submit('track', 10, TargetError(.5, .5))
    loop.tick(10.025)
    assert loop.diagnostics()['read_to_command_ms'] == 25.0
    loop.tick(10.05)
    assert loop.diagnostics()['latency_samples'] == 1
    assert loop.diagnostics()['motor_rate_hz'] == 40.0
    loop.submit('track', 10.05, TargetError(.5, .5))
    loop.tick(10.075)
    assert loop.diagnostics()['latency_samples'] == 2
    loop.tick(10.3)
    assert loop.diagnostics()['latency_samples'] == 2
    assert loop.stale


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


def test_center_during_estop_does_not_execute_after_reset():
    loop = make_loop()
    loop.gimbal.enable()
    loop.gimbal.set_tilt_angle(20)
    loop.stop_motion(emergency=True)
    loop.center()
    loop.tick(10)
    loop.reset_estop()
    loop.resume()
    loop.submit('scan', 10.025, scan_speed=.15)
    loop.tick(10.025)
    assert loop.get_status().commanded_tilt_degrees == 20
    assert not loop._center


def test_reset_discards_pending_center_and_requires_explicit_resume():
    loop = make_loop()
    loop.center()
    loop.reset_estop()
    loop.tick(10)
    assert not loop.get_status().enabled
    assert not loop._center
    assert loop._halt.is_set()


def test_slow_enable_cannot_send_expired_pan_command(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr('software.motor_loop.monotonic', lambda: clock[0])
    loop = make_loop()
    original = loop.gimbal.enable
    def slow_enable():
        clock[0] += .2
        original()
    loop.gimbal.enable = slow_enable
    writes = []
    loop.gimbal.set_pan_speed = writes.append
    loop.submit('scan', 10, scan_speed=.15)
    loop.tick(10.025)
    assert not writes
    assert loop.stale
    assert loop.get_status().commanded_pan_speed == 0


def test_slow_pan_write_holds_and_does_not_send_expired_tilt(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr('software.motor_loop.monotonic', lambda: clock[0])
    loop = make_loop()
    loop.gimbal.enable()
    loop.gimbal.set_tilt_angle(20)
    original = loop.gimbal.set_pan_speed
    def slow_pan(speed):
        original(speed)
        clock[0] += .2
    loop.gimbal.set_pan_speed = slow_pan
    tilts = []
    loop.gimbal.set_tilt_angle = tilts.append
    loop.submit('track', 10, TargetError(1, 1))
    loop.tick(10.025)
    assert not tilts
    assert loop.stale
    assert loop.get_status().commanded_pan_speed == 0
    assert loop.get_status().commanded_tilt_degrees == 20
