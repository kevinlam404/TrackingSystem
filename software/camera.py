from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from threading import Condition, Thread
import cv2
import numpy as np

@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp_ns: int
    sequence: int

class Camera(ABC):
    @abstractmethod
    def start(self) -> None:
        """Start the camera."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame from camera or when none when available."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera."""

class OpenCVCamera(Camera):
    """Camera source for a webcam or precrecorded video file using OpenCV."""
    def __init__(self, source: str | int, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: cv2.VideoCapture | None = None
        self.sequence = 0

    def start(self) -> None:
        if isinstance(self.source, str):
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Video file {self.source} not found.")

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Unable to open camera source {self.source}.")

        if isinstance(self.source, int):
            # Best effort: some webcam backends do not support this setting.
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> Frame | None:
        if self.capture is None:
            raise RuntimeError("Camera has not started")

        success, image = self.capture.read()
        if not success:
            return None
        
        frame = Frame(image=image, timestamp_ns=monotonic_ns(), sequence=self.sequence)
        self.sequence += 1
        return frame

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

class PiCamera(Camera):
    """Camera source for Pi5 Cam 3"""
    def __init__(self,width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.sequence = 0

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError("Picamera is unavailble. Use on Raspberry Pi OS") from error

        self.camera = Picamera2()

        try:
            configuration = self.camera.create_video_configuration(main={"size": (self.width, self.height), "format": "RGB888"}, controls={"FrameRate": self.fps}, buffer_count=2)
            self.camera.configure(configuration)
            self.camera.start()
        except Exception:
            try:
                self.camera.close()
            finally:
                self.camera = None
            raise

    def read(self) -> Frame | None:
        if self.camera is None:
            raise RuntimeError("Camera has not started")
        
        # Picamera2 RGB888 arrays already have OpenCV's B,G,R byte order.
        bgr_image = self.camera.capture_array()

        frame = Frame(image=bgr_image, timestamp_ns=monotonic_ns(), sequence=self.sequence)
        self.sequence += 1
        return frame
        
    def stop(self) -> None:
        if self.camera is not None:
            camera, self.camera = self.camera, None
            try:
                camera.stop()
            finally:
                camera.close()

class LatestFrameCamera(Camera):
    """Captures continuosly and only returns newest frame"""
    def __init__(self, camera: Camera, read_timeout_seconds: float = 0.05) -> None:
        self._camera = camera
        self._read_timeout_seconds = read_timeout_seconds
        self._condition = Condition()
        self._thread: Thread | None = None

        self._latest_frame: Frame | None = None
        self._last_delivered_sequence = -1

        self._running = False
        self._ended = False
        self._error: Exception | None = None

    def start(self) -> None:
        if self._running:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError('Previous camera worker is still stopping')

        with self._condition:
            self._latest_frame = None
            self._last_delivered_sequence = -1
            self._ended = False
            self._error = None
        try:
            self._camera.start()
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
            raise RuntimeError(f'Unable to start camera: {error}') from error
        self._running = True

        self._thread = Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True
        )

        self._thread.start()

    def read(self) -> Frame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._latest_frame is not None 
                    and self._latest_frame.sequence 
                    != self._last_delivered_sequence
                )
                or self._ended 
                or self._error is not None, 
                timeout=self._read_timeout_seconds
            )
            if self._error is not None:
                raise RuntimeError(
                    "Background camera capture failed"
                ) from self._error

            if self._latest_frame is None:
                return None

            if (self._latest_frame.sequence == self._last_delivered_sequence):
                return None

            frame = self._latest_frame
            self._last_delivered_sequence = frame.sequence
            return frame

    @property
    def ended(self) -> bool:
        with self._condition:
            return self._ended

    def wait_for_frame(self, after_sequence: int, timeout: float = 0.1) -> Frame | None:
        """Independent preview reader; does not consume the vision reader's frame."""
        with self._condition:
            self._condition.wait_for(
                lambda: (self._latest_frame is not None and
                         self._latest_frame.sequence != after_sequence) or
                        self._error is not None or self._ended,
                timeout=timeout,
            )
            if self._error is not None:
                raise RuntimeError(f'Camera capture failed: {self._error}') from self._error
            if self._ended:
                raise RuntimeError('Camera stream ended')
            if self._latest_frame is None or self._latest_frame.sequence == after_sequence:
                return None
            return self._latest_frame

    def stop(self) -> None:
        self._running = False
        try:
            self._camera.stop()
        finally:
            with self._condition:
                self._ended = True
                self._condition.notify_all()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
                if self._thread.is_alive():
                    raise RuntimeError('Camera worker did not stop; restart the camera service')
                self._thread = None

    def _capture_loop(self) -> None:
        try:
            while self._running:
                frame = self._camera.read()
                if frame is None:
                    with self._condition:
                        self._ended = True
                        self._condition.notify_all()
                    return

                with self._condition:
                    self._latest_frame = frame
                    self._condition.notify_all()

        except Exception as error:
            if self._running:
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
