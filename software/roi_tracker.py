from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

from software.detector import Detection


class TrackerBackend(Protocol):
    def init(self, image: np.ndarray, box: tuple[int, int, int, int]) -> bool: ...
    def update(self, image: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]: ...


@dataclass(frozen=True)
class ROIMotion:
    velocity_x: float
    velocity_y: float


class ROITracker:
    """Tracks an arbitrary user-selected rectangle and predicts its next center."""

    def __init__(
        self,
        tracker_factory: Callable[[], TrackerBackend] | None = None,
        velocity_smoothing: float = 0.7,
        prediction_frames: float = 1.0,
    ) -> None:
        self._factory = tracker_factory or self._create_backend
        self.velocity_smoothing = max(0.0, min(1.0, velocity_smoothing))
        self.prediction_frames = max(0.0, prediction_frames)
        self._tracker: TrackerBackend | None = None
        self._last_center: tuple[float, float] | None = None
        self._velocity = ROIMotion(0.0, 0.0)
        self.active = False

    @staticmethod
    def _create_backend() -> TrackerBackend:
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        raise RuntimeError(
            "OpenCV CSRT tracker is unavailable; install opencv-contrib-python"
        )

    def start(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int],
    ) -> Detection:
        x, y, width, height = roi
        if width < 2 or height < 2:
            raise ValueError("ROI must have positive width and height")
        tracker = self._factory()
        initialized = tracker.init(image, (x, y, width, height))
        if initialized is False:
            raise RuntimeError("Unable to initialize ROI tracker")
        self._tracker = tracker
        self._last_center = (x + width / 2.0, y + height / 2.0)
        self._velocity = ROIMotion(0.0, 0.0)
        self.active = True
        return self._detection(x, y, width, height)

    def update(self, image: np.ndarray) -> Detection | None:
        if not self.active or self._tracker is None:
            return None
        success, box = self._tracker.update(image)
        if not success:
            self.clear()
            return None
        x, y, width, height = box
        current_center = (x + width / 2.0, y + height / 2.0)
        if self._last_center is not None:
            measured_x = current_center[0] - self._last_center[0]
            measured_y = current_center[1] - self._last_center[1]
            keep = self.velocity_smoothing
            self._velocity = ROIMotion(
                keep * self._velocity.velocity_x + (1.0 - keep) * measured_x,
                keep * self._velocity.velocity_y + (1.0 - keep) * measured_y,
            )
        self._last_center = current_center
        predicted_x = x + self._velocity.velocity_x * self.prediction_frames
        predicted_y = y + self._velocity.velocity_y * self.prediction_frames
        return self._detection(predicted_x, predicted_y, width, height)

    def clear(self) -> None:
        self._tracker = None
        self._last_center = None
        self._velocity = ROIMotion(0.0, 0.0)
        self.active = False

    @staticmethod
    def _detection(
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> Detection:
        return Detection(
            x1=int(round(x)),
            y1=int(round(y)),
            x2=int(round(x + width)),
            y2=int(round(y + height)),
            confidence=1.0,
            class_id=-1,
            label="manual_roi",
            track_id=-1,
        )
