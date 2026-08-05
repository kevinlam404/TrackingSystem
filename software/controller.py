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
    

            