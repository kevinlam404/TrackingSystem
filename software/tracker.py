from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto 
import cv2
import numpy as np
from software.detector import Detection

class TrackingState(Enum):
    IDLE = auto()
    TRACKING = auto()
    SEARCHING = auto()
    LOST = auto()

@dataclass(frozen=True)
class TrackedTarget:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.x1 + self.width // 2,
            self.y1 + self.height // 2
        )

class ObjectTracker:
    def __init__(self) -> None:
        self.state = TrackingState.IDLE
        self._tracker = None
        self._target: TrackedTarget | None = None

    @property
    def target(self) -> TrackedTarget | None:
        return self.target

    def search(self) -> None:
        self._tracker = None
        self._target = None
        self.state = TrackingState.SEARCHING

    def start(self, frame:ndarray, detection: Detection) -> None:
        self._tracker = self.create_kcf_tracker()
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        bounding_box = (detection.x1, detection.y1, width, height)
        self._tracker.init(frame, bounding_box)

        self._target = TrackedTarget(
            x1=detection.x1,
            y1=detection.y1,
            x2=detection.x2,
            y2=detection.y2,
            confidence=detection.confidence,
            label=detection.label   
        )

        self.state = TrackingState.TRACKING
        return self._target

    def update(self, frame:np.ndarray) -> TrackedTarget | None:
        if self._tracker is None:
            return None

        success, bounding_box = self._tracker.update(frame)

        if not success:
            self._tracker = None
            self._target = None
            self.state = TrackingState.LOST
            return None

        x, y, width, height = bounding_box
        frame_height, frame_width = frame.shape[:2]

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(frame_width - 1, int(x + width))
        y2 = min(frame_height - 1, int(y + height))

        if x2 <= x1 or y2 <= y1:
            self._tracker = None
            self._target = None
            self.state = TrackingState.LOST
            return None

        previous = self._target

        self._target = TrackedTarget(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=previous.confidence if previous else 0.0,
            label=previous.label if previous else ""
        )

        self.state = TrackingState.TRACKING
        return self._target

    def reset(self) -> None:
        self._tracker = None
        self._target = None
        self.state = TrackingState.IDLE

    @staticmethod
    def select_largest_person(detections: list[Detection]) -> Detection | None:
        people = [detection for detection in detections if detection.label.lower() == "person"]
        if not people:
            return None

        return max(people, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))

    @staticmethod
    def create_kcf_tracker():
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()

        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerKCF_create()

        raise RuntimeError("KCF is unavailable.")

        

