from threading import Event
from time import monotonic

from software.controller import OpenLoopTrackingController, TargetError
from software.gimbal import SimulatedGimbal
from software.motor_loop import MotorLoop


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
