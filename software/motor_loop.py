"""Single owner of servo I/O, independent of camera/inference cadence."""
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic

from software.controller import OpenLoopTrackingController, TargetError


@dataclass(frozen=True)
class MotorIntent:
    mode: str = 'idle'
    observed_at: float = 0.0
    error: TargetError | None = None
    scan_speed: float = 0.0


class MotorLoop:
    def __init__(self, gimbal, controller: OpenLoopTrackingController,
                 rate_hz=40.0, stale_seconds=0.5):
        if not 1 <= rate_hz <= 100 or stale_seconds <= 0:
            raise ValueError('Invalid motor rate or stale timeout')
        self.gimbal, self.controller = gimbal, controller
        self.period, self.stale_seconds = 1.0 / rate_hz, stale_seconds
        self._lock = Lock()
        self._intent = MotorIntent()
        self._stop = Event()
        self._thread = None
        self._last_tick = monotonic()
        self._status = gimbal.get_status()
        self.fault = None
        self.stale = True
        self.ticks = 0
        self._estop = False
        self._center = False
        self._paused = False

    def submit(self, mode, observed_at, error=None, scan_speed=0.0):
        if mode not in ('idle', 'hold', 'scan', 'track'):
            raise ValueError('Unknown motor intent')
        with self._lock:
            if not self._paused:
                self._intent = MotorIntent(mode, observed_at, error, scan_speed)

    def resume(self):
        with self._lock:
            self._paused = False

    def stop_motion(self, emergency=False):
        with self._lock:
            self._intent = MotorIntent()
            self._estop = self._estop or emergency
            self._center = False
            self._paused = True

    def reset_estop(self):
        with self._lock:
            self._intent = MotorIntent()
            self._estop = False
            self._paused = True

    def center(self):
        with self._lock:
            self._center = True
            self._intent = MotorIntent()

    def get_status(self):
        with self._lock:
            return self._status

    def diagnostics(self):
        with self._lock:
            return {
                'motor_stale': self.stale,
                'motor_ticks': self.ticks,
                'emergency_stopped': self._estop,
                'gimbal_enabled': self._status.enabled,
                'pan_speed': round(self._status.commanded_pan_speed, 3),
                'tilt_degrees': round(self._status.commanded_tilt_degrees, 2),
                'fault': self.fault or self._status.fault,
            }

    def tick(self, now=None):
        now = monotonic() if now is None else now
        # Hold lock across I/O: stop cannot be overwritten by a tick using old intent.
        with self._lock:
            intent = self._intent
            dt = max(0.0, min(0.1, now - self._last_tick))
            self._last_tick = now
            self.stale = now - intent.observed_at >= self.stale_seconds or now < intent.observed_at
            try:
                status = self.gimbal.get_status()
                if self._estop or self.fault or intent.mode == 'idle' and not self._center:
                    if status.enabled:
                        self.gimbal.disable()
                elif self._center:
                    if not status.enabled:
                        self.gimbal.enable()
                    self.gimbal.center()
                    self._center = False
                    # Keep the centered tilt powered instead of disabling it
                    # on the next idle tick before the servo has moved.
                    self._intent = MotorIntent('hold', now)
                elif self.stale or intent.mode == 'hold':
                    if status.enabled:
                        self.gimbal.hold()
                else:
                    if not status.enabled:
                        self.gimbal.enable()
                    if intent.mode == 'scan':
                        self.gimbal.set_pan_speed(intent.scan_speed)
                    elif intent.error is not None:
                        command = self.controller.update(intent.error, status.commanded_tilt_degrees, dt)
                        self.gimbal.set_pan_speed(command.pan_speed)
                        self.gimbal.set_tilt_angle(command.tilt_degrees)
                    else:
                        self.gimbal.hold()
            except Exception as error:
                self.fault = str(error)
                try:
                    self.gimbal.disable()
                except Exception:
                    pass
            self._status = self.gimbal.get_status()
            self.ticks += 1

    def start(self):
        self._thread = Thread(target=self._run, name='motor-control', daemon=True)
        self._thread.start()

    def _run(self):
        deadline = monotonic()
        while not self._stop.is_set():
            self.tick()
            deadline += self.period
            if deadline < monotonic():
                deadline = monotonic() + self.period
            self._stop.wait(max(0.0, deadline - monotonic()))
        with self._lock:
            try:
                self.gimbal.disable()
            except Exception as error:
                self.fault = str(error)

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError('Motor I/O did not stop; use physical power switch')
