#converts the selected bounding boxes center into a normalized pan and tilt error
#select target center -> compare camera center -> pan and tilt error -> eventually servo movement

from __future__ import annotations
from dataclasses import dataclass
from software.detector import Detection

@dataclass(frozen=True)
class TargetError:
    pan: float
    tilt: float

    @property
    def horizontal_direction(self) -> str:
        if self.pan > 0:
            return "RIGHT"

        if self.pan < 0:
            return "LEFT"

        return "CENTER"

    @property
    def vertical_direction(self) -> str:
        if self.tilt > 0:
            return "UP"

        if self.tilt < 0:
            return "DOWN"

        return "CENTER"

def calculate_target_error(target: Detection, frame_width: int, frame_height:int, dead_zone: float = 0.05) -> TargetError:
    target_x, target_y = target.center
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    #Right = positive
    pan_error = (target_x - frame_center_x) / frame_center_x
    #Up = positive
    tilt_error = (frame_center_y - target_y) / frame_center_y

    pan_error = max(-1.0, min(1.0, pan_error))
    tilt_error = max(-1.0, min(1.0, tilt_error))

    if abs(pan_error) < dead_zone:
        pan_error = 0.0

    if abs(tilt_error) < dead_zone:
        tilt_error = 0.0

    return TargetError(pan=pan_error, tilt=tilt_error)

@dataclass(frozen=True)
class MotionCommand:
    pan_speed: float
    tilt_degrees: float

class OpenLoopTrackingController:
    def __init__(self, maximum_pan_speed: float=0.4,minimum_pan_speed: float=0.08,tilt_speed_dps:float = 45.0,tilt_min_degrees: float=-80.0, tilt_max_degrees: float=80.0) -> None:
        self.maximum_pan_speed = maximum_pan_speed
        self.minimum_pan_speed = minimum_pan_speed
        self.tilt_speed_dps = tilt_speed_dps
        self.tilt_min_degrees = tilt_min_degrees
        self.tilt_max_degrees = tilt_max_degrees

    def update(self, target_error: TargetError, current_tilt_degrees: float, elapsed_seconds: float) -> MotionCommand:
        elapsed_seconds = max(0.0, min(0.1,elapsed_seconds))
        pan_speed = self._calculate_pan_speed(target_error.pan)
        tilt_change = (target_error.tilt * self.tilt_speed_dps * elapsed_seconds)
        tilt_degrees = self._clamp(current_tilt_degrees + tilt_change, self.tilt_min_degrees, self.tilt_max_degrees)
        
        return MotionCommand(pan_speed=pan_speed, tilt_degrees=tilt_degrees)
    
    def _calculate_pan_speed(self, pan_error) -> float:
        if pan_error == 0.0:
            return 0.0

        speed = abs(pan_error) * self.maximum_pan_speed
        speed = max(self.minimum_pan_speed, speed)
        speed = min(self.maximum_pan_speed, speed)

        if pan_error < 0:
            speed = -speed

        return speed

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

            