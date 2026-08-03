from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
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
    """Camera source for a webcame or precrecorded video file using OpenCV."""
    def __init__(self, source: str | int, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.capture: cv2.VideoCapture | None = None
        self.sequence = 0

    def start(self) -> None:
        if isinstance(self.source, int):
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"Video file {self.source} not found.")

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Unable to open camera source {self.source}.")

        if isinstance(self.source, int):
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
        self.capture: cv2.VideoCapture | None = None
        self.sequence = 0

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError("Picamera is unavailble. Use on Raspberry Pi OS") from error

        self.camera = Picamera2()

        configuration = self.camera.create_video_configuration(main={"size": (self.width, self.height), "format": "RGB888"}, controls={"FrameRate": self.fps}, buffer_count=2)
        self.camera.configure(configuration)
        self.camera.start()

    def read(self) -> Frame | None:
        if self.camera is None:
            raise RuntimeError("Camera has not started")
        
        rgb_image = self.camera.capture_array()
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        frame = Frame(image=bgr_image, timestamp_ns=monotonic_ns(), sequence=self.sequence)
        self.sequence += 1
        return frame
        
    def stop(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None