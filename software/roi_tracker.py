from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
from time import monotonic

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
        prediction_seconds: float = 0.05,
        reacquire_timeout_seconds: float = 1.0,
        appearance_threshold: float = 0.8,
    ) -> None:
        self._factory = tracker_factory or self._create_backend
        self.velocity_smoothing = max(0.0, min(1.0, velocity_smoothing))
        self._tracker: TrackerBackend | None = None
        self._last_center: tuple[float, float] | None = None
        self._velocity = ROIMotion(0.0, 0.0)
        self.active = False
        self.prediction_seconds = prediction_seconds
        self.reacquire_timeout_seconds = max(0.0, reacquire_timeout_seconds)
        self.appearance_threshold = appearance_threshold
        self.state = 'IDLE'
        self._timestamp = None
        self._box = None
        self._template = None
        self._lost_since = None

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
        timestamp: float | None = None,
    ) -> Detection:
        x, y, width, height = roi
        if width < 2 or height < 2:
            raise ValueError("ROI must have positive width and height")
        frame_height, frame_width = image.shape[:2]
        if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
            raise ValueError("ROI must stay inside the camera image")
        tracker = self._factory()
        initialized = tracker.init(image, (x, y, width, height))
        if initialized is False:
            raise RuntimeError("Unable to initialize ROI tracker")
        self._tracker = tracker
        self._last_center = (x + width / 2.0, y + height / 2.0)
        self._velocity = ROIMotion(0.0, 0.0)
        self.active = True
        self.state = 'LOCKED'
        self._timestamp = monotonic() if timestamp is None else timestamp
        self._box = roi
        self._lost_since = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        patch = gray[y:y + height, x:x + width]
        self._template = cv2.resize(patch, (min(width, 96), min(height, 96)))
        # Flat patches have no reliable appearance signature for reacquisition.
        if self._template.std() < 3:
            self._template = None
        return self._detection(x, y, width, height)

    def update(self, image: np.ndarray, timestamp: float | None = None) -> Detection | None:
        if not self.active or self._tracker is None:
            return None
        now = monotonic() if timestamp is None else timestamp
        dt = max(0.001, now - self._timestamp)
        if self._lost_since is not None:
            return self._reacquire(image, now)
        try:
            success, box = self._tracker.update(image)
        except cv2.error:
            success, box = False, self._box
        if success:
            x, y, width, height = box
            fh, fw = image.shape[:2]
            success = (np.isfinite(box).all() and width >= 2 and height >= 2
                       and x >= 0 and y >= 0 and x + width <= fw and y + height <= fh)
            if success and self._template is not None:
                patch = image[int(y):int(y + height), int(x):int(x + width)]
                patch = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY),
                                   (self._template.shape[1], self._template.shape[0]))
                score = float(cv2.matchTemplate(patch, self._template, cv2.TM_CCOEFF_NORMED)[0, 0])
                # CSRT is the authority for an ongoing track. Template correlation
                # drops on rotation/lighting changes and must NOT veto its result.
                # Refresh appearance only on a strong match, for later recovery.
                if np.isfinite(score) and score >= 0.7:
                    self._template = cv2.addWeighted(self._template, 0.95, patch, 0.05, 0)
        if not success:
            self._lost_since = now
            self.state = 'REACQUIRING'
            return None
        x, y, width, height = box
        current_center = (x + width / 2.0, y + height / 2.0)
        if self._last_center is not None:
            measured_x = (current_center[0] - self._last_center[0]) / dt
            measured_y = (current_center[1] - self._last_center[1]) / dt
            keep = self.velocity_smoothing
            self._velocity = ROIMotion(
                keep * self._velocity.velocity_x + (1.0 - keep) * measured_x,
                keep * self._velocity.velocity_y + (1.0 - keep) * measured_y,
            )
        self._last_center = current_center
        self._timestamp = now
        self._box = box
        horizon = min(0.15, max(0.0, self.prediction_seconds))
        # Bound prediction to half a box: a bad velocity cannot fling the motors.
        predicted_x = np.clip(x + np.clip(self._velocity.velocity_x * horizon, -width / 2, width / 2), 0, image.shape[1] - width)
        predicted_y = np.clip(y + np.clip(self._velocity.velocity_y * horizon, -height / 2, height / 2), 0, image.shape[0] - height)
        return self._detection(predicted_x, predicted_y, width, height)

    def _reacquire(self, image, now):
        if now - self._lost_since > self.reacquire_timeout_seconds:
            self.clear()
            self.state = 'LOST'
            return None
        if self._template is None:
            return None
        x, y, width, height = self._box
        fh, fw = image.shape[:2]
        # Local search only; not global identity recognition. Cap compute on Pi.
        margin = min(120, 2 * max(width, height))
        left, top = max(0, int(x - margin)), max(0, int(y - margin))
        right, bottom = min(fw, int(x + width + margin)), min(fh, int(y + height + margin))
        gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        factor = min(1.0, 96.0 / max(width, height))
        gray = cv2.resize(gray, None, fx=factor, fy=factor)
        best = None
        for scale in (0.9, 1.0, 1.1):
            tw, th = max(2, round(width * factor * scale)), max(2, round(height * factor * scale))
            if tw > gray.shape[1] or th > gray.shape[0]:
                continue
            template = cv2.resize(self._template, (tw, th))
            scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            # Reject ambiguous matches (another similar object nearby).
            others = scores.copy()
            px, py = location
            others[max(0, py-th//2):py+th//2+1, max(0, px-tw//2):px+tw//2+1] = -1
            if score >= self.appearance_threshold and score - float(others.max()) >= 0.08:
                if best is None or score > best[0]:
                    best = (score, (left + round(px / factor), top + round(py / factor), round(tw / factor), round(th / factor)))
        if best is None:
            return None
        try:
            return self.start(image, best[1], timestamp=now)
        except (ValueError, RuntimeError, cv2.error):
            return None

    def clear(self) -> None:
        self._tracker = None
        self._last_center = None
        self._velocity = ROIMotion(0.0, 0.0)
        self.active = False
        self.state = 'IDLE'
        self._lost_since = None
        self._template = None

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
