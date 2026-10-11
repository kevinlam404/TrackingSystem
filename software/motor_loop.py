"""Single owner of servo I/O, independent of camera/inference cadence."""
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from math import isfinite
from collections import deque

from software.controller import OpenLoopTrackingController, TargetError


@dataclass(frozen=True)
class MotorIntent:
    mode: str = 'idle'
    observed_at: float = 0.0
    error: TargetError | None = None
    scan_speed: float = 0.0


class MotorLoop:
    def __init__(self, gimbal, controller: OpenLoopTrackingController,
                 rate_hz=40.0, stale_seconds=0.5, pan_acceleration=1.0):
        if not 1 <= rate_hz <= 100 or stale_seconds <= 0:
            raise ValueError('Invalid motor rate or stale timeout')
        if not 0 < pan_acceleration <= 20:
            raise ValueError('Pan acceleration must be in (0, 20] normalized speed/s')
        self.pan_acceleration = pan_acceleration
        self._io_lock = Lock()
        self._halt = Event()
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
        self._generation = 0
        self._tick_times = deque(maxlen=200)
        self._command_latencies = deque(maxlen=200)
        self._last_measured_observation = None
        self._tick_duration_ms = 0.0

    def generation(self):
        with self._lock:
            return self._generation

    def submit(self, mode, observed_at, error=None, scan_speed=0.0):
        if mode not in ('idle', 'hold', 'scan', 'track'):
            raise ValueError('Unknown motor intent')
        values = [observed_at, scan_speed]
        if error is not None:
            values += [error.pan, error.tilt]
        if not all(isfinite(value) for value in values):
            raise ValueError('Motor intent must contain finite values')
        with self._lock:
            if not self._paused:
                self._intent = MotorIntent(mode, observed_at, error, scan_speed)

    def resume(self, expected_generation=None):
        with self._lock:
            if expected_generation is not None and expected_generation != self._generation:
                return False
            if self._estop or self.fault:
                return False
            self._paused = False
            self._halt.clear()
            return True

    def stop_motion(self, emergency=False):
        with self._lock:
            self._generation += 1
            self._halt.set()
            self._intent = MotorIntent()
            self._estop = self._estop or emergency
            self._center = False
            self._paused = True

    def reset_estop(self):
        with self._lock:
            self._generation += 1
            self._halt.set()
            self._center = False
            self._intent = MotorIntent()
            self._estop = False
            self._paused = True

    def center(self):
        with self._lock:
            self._generation += 1
            if self._estop or self.fault:
                return
            self._center = True
            self._intent = MotorIntent()
            self._paused = True
            if not self._estop:
                self._halt.clear()

    def get_status(self):
        with self._lock:
            return self._status

    def diagnostics(self):
        with self._lock:
            samples = sorted(self._command_latencies)
            rate = ((len(self._tick_times)-1)/(self._tick_times[-1]-self._tick_times[0])
                    if len(self._tick_times) > 1 and self._tick_times[-1] > self._tick_times[0] else 0.0)
            return {
                'motor_rate_hz': round(rate, 1),
                'motor_tick_duration_ms': round(self._tick_duration_ms, 1),
                'read_to_command_ms': round(self._command_latencies[-1], 1) if samples and not self.stale and self._status.enabled and not self._halt.is_set() else None,
                'read_to_command_p95_ms': round(samples[min(len(samples)-1, int(.95*len(samples)))], 1) if samples else None,
                'latency_samples': len(samples),
                'motor_stale': self.stale,
                'motor_ticks': self.ticks,
                'emergency_stopped': self._estop,
                'gimbal_enabled': self._status.enabled,
                'pan_speed': round(self._status.commanded_pan_speed, 3),
                'tilt_degrees': round(self._status.commanded_tilt_degrees, 2),
                'fault': self.fault or self._status.fault,
            }

    def tick(self, now=None):
        started_at = monotonic()
        now = started_at if now is None else now
        # Preserve deterministic tick(now) tests while accounting for time spent
        # inside driver calls. Never treat the tick's start time as still current.
        def current_time():
            return now + max(0.0, monotonic() - started_at)

        def fresh():
            self.stale = self._expired(intent, current_time())
            if self._halt.is_set():
                return False
            if self.stale:
                self.gimbal.hold()
                return False
            return True
        # The intent lock never spans I2C: stop and diagnostics remain responsive.
        # Only this loop owns hardware I/O. A blocked bus still needs power cutoff.
        with self._io_lock:
            with self._lock:
                intent = self._intent
                center = self._center
            dt = max(0.0, min(0.1, now - self._last_tick))
            self._last_tick = now
            self.stale = self._expired(intent, current_time())
            try:
                status = self.gimbal.get_status()
                self.stale = self._expired(intent, current_time())
                if self._halt.is_set() or self.fault or intent.mode == 'idle' and not center:
                    if status.enabled:
                        self.gimbal.disable()
                elif center:
                    if not status.enabled:
                        self.gimbal.enable()
                    if self._halt.is_set():
                        self.gimbal.disable()
                    else:
                        self.gimbal.hold()
                        if not self._halt.is_set():
                            self.gimbal.set_tilt_angle(0.0)
                    # Keep the centered tilt powered instead of disabling it
                    # on the next idle tick before the servo has moved.
                    with self._lock:
                        self._center = False
                        if not self._halt.is_set():
                            self._intent = MotorIntent('hold', now)
                elif self.stale or intent.mode == 'hold':
                    if status.enabled:
                        self.gimbal.hold()
                else:
                    if not status.enabled:
                        self.gimbal.enable()
                    if self._halt.is_set():
                        self.gimbal.disable()
                    elif intent.mode == 'scan' and fresh():
                        self._set_ramped_pan(intent.scan_speed, status.commanded_pan_speed, dt)
                    elif intent.error is not None and fresh():
                        command = self.controller.update(intent.error, status.commanded_tilt_degrees, dt)
                        if fresh():
                            self._set_ramped_pan(command.pan_speed, status.commanded_pan_speed, dt)
                        if fresh():
                            self.gimbal.set_tilt_angle(command.tilt_degrees)
                    else:
                        self.gimbal.hold()
                if intent.mode in ('track', 'scan') and self.gimbal.get_status().enabled:
                    # If a movement write itself was slow, neutralize as soon as
                    # it returns. No software can cancel an in-flight I2C write.
                    fresh()
                if self._halt.is_set() and self.gimbal.get_status().enabled:
                    self.gimbal.disable()
            except Exception as error:
                self.fault = str(error)
                try:
                    self.gimbal.disable()
                except Exception:
                    pass
            try:
                status = self.gimbal.get_status()
            except Exception as error:
                self.fault = str(error)
                status = self._status
            with self._lock:
                finished_at = current_time()
                self._tick_times.append(now)
                self._tick_duration_ms = (monotonic()-started_at)*1000
                if (intent.mode in ('track', 'scan') and not self.stale
                        and not self._halt.is_set() and not self.fault and status.enabled
                        and intent.observed_at != self._last_measured_observation):
                    self._command_latencies.append(max(0.0, finished_at-intent.observed_at)*1000)
                    self._last_measured_observation = intent.observed_at
                self._status = status
                self.ticks += 1

    def _expired(self, intent, now):
        return now < intent.observed_at or now - intent.observed_at >= self.stale_seconds

    def _set_ramped_pan(self, requested, current, dt):
        requested = max(-1.0, min(1.0, requested))
        change = self.pan_acceleration * dt
        speed = max(current - change, min(current + change, requested))
        if not self._halt.is_set():
            self.gimbal.set_pan_speed(speed)

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
        with self._io_lock:
            try:
                self.gimbal.disable()
            except Exception as error:
                self.fault = str(error)

    def close(self):
        self._halt.set()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError('Motor I/O did not stop; use physical power switch')
