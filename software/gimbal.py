from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class GimbalStatus:
    enabled: bool
    commanded_pan_speed: float #commanded not measured speed
    commanded_tilt_degrees: float #commanded not measured tilt
    fault: str | None = None

class Gimbal(ABC):
    @abstractmethod
    def enable(self) -> None:
        """Enable movement commands."""

    @abstractmethod
    def disable(self) -> None:
        """Stop pan and reject movement commands."""

    @abstractmethod
    def set_pan_speed(self, speed:float) -> None:
        """Set pan speed from -1.0 to +1.0."""

    @abstractmethod
    def set_tilt_angle(self, degrees:float) -> None:
        """Set zero-centered tilt angle."""

    @abstractmethod
    def hold(self) -> None:
        """Stop pan while maintaining tilt."""

    @abstractmethod
    def center(self) -> None:
        """Stop pan and center tilt."""

    @abstractmethod
    def get_status(self) -> GimbalStatus:
        """Return commanded gimbal state."""

class SimulatedGimbal(Gimbal):
    def __init__(self, tilt_min_degrees: float = -80.0, tilt_max_degrees: float = 80.0) -> None:
        self.tilt_min_degrees = tilt_min_degrees
        self.tilt_max_degrees = tilt_max_degrees
        
        self._enabled = False
        self._pan_speed = 0.0
        self._tilt_degrees = 0.0

    def enable(self) -> None:
        self._pan_speed = 0.0
        self._enabled = True

    def disable(self) -> None:
        self._pan_speed = 0.0
        self._enabled = False

    def set_pan_speed(self, speed: float) -> None:
        self._require_enabled()
        self._pan_speed = self._clamp(speed,-1.0,1.0)

    def set_tilt_angle(self, degrees:float) -> None:
        self._require_enabled()
        self._tilt_degrees = self._clamp(
            degrees,
            self.tilt_min_degrees,
            self.tilt_max_degrees
        )

    def hold(self) -> None:
        self._pan_speed = 0.0

    def center(self) -> None:
        self._require_enabled()
        self._pan_speed = 0.0
        self._tilt_degrees = 0.0

    def get_status(self) -> GimbalStatus:
        return GimbalStatus(
            enabled=self._enabled,
            commanded_pan_speed=self._pan_speed,
            commanded_tilt_degrees=self._tilt_degrees
        )

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise RuntimeError("Gimbal is disabled")

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum,value))

class PCA9685Gimbal(Gimbal):
    def __init__(
        self,
        *,
        pan_channel:int,
        tilt_channel:int,
        pan_min_pulse_us:int,
        pan_stop_pulse_us:int,
        pan_max_pulse_us:int,
        tilt_min_pulse_us:int,
        tilt_max_pulse_us:int,
        tilt_min_degrees: float = -80.0,
        tilt_max_degrees: float = 80.0,
        pan_reversed: bool = False,
        tilt_reversed: bool = False,
        address: int = 0x40,
    ) -> None:
        try:
            from adafruit_servokit import ServoKit
        except ImportError as error:
            raise RuntimeError("Install adafruit-circuitpython-servokit") from error
        if pan_channel == tilt_channel:
            raise ValueError("Pan and tilt cannot use the same channel")
        if not (
            pan_min_pulse_us < pan_stop_pulse_us < pan_max_pulse_us
        ):
            raise ValueError("Pan pulse widths must be min < stop < max")


        self.tilt_min_degrees = tilt_min_degrees
        self.tilt_max_degrees = tilt_max_degrees

        self.pan_reversed = pan_reversed
        self.tilt_reversed = tilt_reversed
        self.pan_stop_throttle = self._pulse_to_throttle(
            pulse_us=pan_stop_pulse_us,
            minimum_pulse_us=pan_min_pulse_us,
            maximum_pulse_us=pan_max_pulse_us
        )

        self._kit = ServoKit(
            channels=16,
            address=address,
            frequency=50,
        )
        self._pan_servo = self._kit.continuous_servo[pan_channel]
        self._tilt_servo = self._kit.servo[tilt_channel]
        self._pan_servo.set_pulse_width_range(min_pulse=pan_min_pulse_us,max_pulse=pan_max_pulse_us)
        self._tilt_servo.actuation_range = 180
        self._tilt_servo.set_pulse_width_range(min_pulse=tilt_min_pulse_us,max_pulse=tilt_max_pulse_us)
        
        self._enabled=False
        self._pan_speed = 0.0
        self._tilt_degrees = 0.0
        self._fault: str | None = None
        #stop continuous pan immediately at startup
        self._pan_servo.throttle = self.pan_neutral_trim

    @staticmethod
    def _pulse_to_throttle(pulse_us:int, minimum_pulse_us:int, maximum_pulse_us:int) -> float:
        midpoint_us = (mimum_pulse_us + maximum_pulse_us) / 2.0
        if pulse_us >= midpoint_us:
            available_range = maximum_pulse_us - midpoint_us
            return (pulse_us - midpoint_us) / available_range

        available_range = midpoint_us - minimum_pulse_us
        return (pulse_us - midpoint_us) / available_range

    def _pan_speed_to_throttle(self, speed: float) -> float:
        speed = self._clamp(speed,-1.0,1.0)
        stop = self.pan_stop_throttle
        if speed >= 0.0:
            return stop + speed * (1.0 - stop)
        if speed < 0.0:
            return stop + speed * (stop + 1.0)
        return stop
        

    def enable(self) -> None:
        self._fault = None
        self._pan_speed = 0.0
        #enabling never starts rotation
        self._pan_servo.throttle = self.pan_neutral_trim
        self._enabled = True
        

    def disable(self) -> None:
        try:
            self._pan_speed = 0.0
            self._pan_servo.throttle = self.pan_neutral_trim
        finally:
            self._enabled = False

    def set_pan_speed(self,speed:float) -> None:
        self._require_enabled()
        user_command = self._clamp(speed,-1.0,1.0)
        hardware_command = user_command
        if self.pan_reversed:
            hardware_command = -hardware_command

        if abs(hardware_command) < 0.01:
            hardware_throttle = self.pan_neutral_trim
        else:
            hardware_throttle = self._clamp(hardware_command + self.pan_neutral_trim,-1.0,1.0)

        try:
            self._pan_servo.throttle = hardware_throttle
            #store user-facing command before reversal
            self._pan_speed = user_command
        except Exception as error:
            self._set_fault(error)
            raise

    def set_tilt_angle(self, degrees: float) -> None:
        self._require_enabled()
        commanded_degrees = self._clamp(degrees, self.tilt_min_degrees,self.tilt_max_degrees)
        hardware_degrees = commanded_degrees
        if self.tilt_reversed:
            hardware_degrees = -hardware_degrees
        #convert software -90..+90 to servo 0..180
        servo_angle = hardware_degrees + 90.0
        servo_angle = self._clamp(servo_angle, 0.0,180.0)

        try:
            self._tilt_servo.angle = servo_angle
            self._tilt_degrees = commanded_degrees
        except Exception as error:
            self._set_fault(error)
            raise

    def hold(self) -> None:
        try:
            self._pan_speed = 0.0
            self._pan_servo.throttle = self.pan_neutral_trim
        except Exception as error:
            self._set_fault(error)
            raise

    def center(self) -> None:
        self._require_enabled()
        self.hold()
        self.set_tilt_angle(0.0)

    def get_status(self) -> GimbalStatus:
        return GimbalStatus(
            enabled=self._enabled,
            commanded_pan_speed = self._pan_speed,
            commanded_tilt_degrees= self._tilt_degrees,
            fault=self._fault
        )

    def _set_fault(self,error:Exception) -> None:
        self._fault = str(error)
        self._enabled = False
        self._pan_speed = 0.0
        try:
            self._pan_servo.throttle = self.pan_neutral_trim
        except Exception:
            pass

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise RuntimeError("Gimbal is disabled")

        if self._fault is not None:
            raise RuntimeError(f"Gimbal hardware fault: {self._fault}")

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))