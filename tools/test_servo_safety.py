"""Hardware-free checks of calibration and PCA9685 command semantics."""
from types import SimpleNamespace
import sys

import pytest

from software.gimbal import PCA9685Gimbal
from tools import servo_calibration as calibration


@pytest.mark.parametrize('pulse', [899, 2101, -100, 20000])
def test_conversion_rejects_unsafe_pulses(pulse):
    with pytest.raises(ValueError):
        calibration.pulse_us_to_duty_cycle(pulse)


@pytest.mark.parametrize('pulse, step', [(899, 5), (2101, 5), (1500, 0), (1500, -1)])
def test_invalid_startup_rejected_before_hardware_access(monkeypatch, pulse, step):
    monkeypatch.setattr(calibration, 'parse_args', lambda: SimpleNamespace(
        channel=0, address=64, start_pulse=pulse, step=step))
    with pytest.raises(ValueError):
        calibration.main()


def test_calibration_all_commands_stay_bounded(monkeypatch):
    class Channel:
        writes = []
        @property
        def duty_cycle(self):
            return self.writes[-1]
        @duty_cycle.setter
        def duty_cycle(self, value):
            self.writes.append(value)
    channel = Channel()
    deinitialized = []
    pca = SimpleNamespace(channels=[channel], deinit=lambda: deinitialized.append(True))
    monkeypatch.setitem(sys.modules, 'board', SimpleNamespace(SCL=1, SDA=2))
    monkeypatch.setitem(sys.modules, 'busio', SimpleNamespace(I2C=lambda *a: object()))
    monkeypatch.setitem(sys.modules, 'adafruit_pca9685', SimpleNamespace(PCA9685=lambda *a, **kw: pca))
    monkeypatch.setattr(calibration, 'parse_args', lambda: SimpleNamespace(channel=0, address=64, start_pulse=1500, step=5))
    commands = iter(['on', 'pulse 2100', '+', 'pulse 900', '-',
                     'step bad', 'step -1', 'pulse bad', 'pulse 2101', 'quit'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(commands))
    calibration.main()
    assert channel.writes == [0, calibration.pulse_us_to_duty_cycle(1500),
                              calibration.pulse_us_to_duty_cycle(2100),
                              calibration.pulse_us_to_duty_cycle(900), 0]
    assert deinitialized == [True]


def driver():
    gimbal = object.__new__(PCA9685Gimbal)
    gimbal._enabled = True
    gimbal._pan_speed = 0.4
    gimbal._tilt_degrees = 20
    gimbal._fault = None
    gimbal.pan_stop_throttle = 0.02
    gimbal._pan_servo = SimpleNamespace(throttle=0.4)
    gimbal._tilt_servo = SimpleNamespace(angle=110)
    return gimbal


def test_disable_keeps_neutral_and_tilt_torque():
    gimbal = driver()
    gimbal.disable()
    assert gimbal._pan_servo.throttle == 0.02
    assert gimbal._tilt_servo.angle == 110
    assert not gimbal.get_status().enabled
    with pytest.raises(RuntimeError):
        gimbal.set_pan_speed(0.4)


def test_failed_stop_still_latches_disabled_and_fault():
    class FailedServo:
        @property
        def throttle(self):
            return 0.4
        @throttle.setter
        def throttle(self, value):
            raise OSError('bus unavailable')
    gimbal = driver()
    gimbal._pan_servo = FailedServo()
    with pytest.raises(OSError):
        gimbal.disable()
    assert not gimbal.get_status().enabled
    assert gimbal.get_status().fault == 'bus unavailable'


@pytest.mark.parametrize('override', [dict(pan_channel=16), dict(tilt_min_pulse_us=2100),
                                    dict(tilt_max_degrees=100)])
def test_invalid_driver_configuration_rejected_before_import(override):
    config = dict(pan_channel=0, tilt_channel=1, pan_min_pulse_us=1000,
                  pan_stop_pulse_us=1500, pan_max_pulse_us=2000,
                  tilt_min_pulse_us=1000, tilt_max_pulse_us=2000)
    config.update(override)
    with pytest.raises(ValueError):
        PCA9685Gimbal(**config)
