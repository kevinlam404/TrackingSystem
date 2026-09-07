from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO

@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str
    track_id: int | None = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property 
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int,int]:
        return(
            self.x1 + self.width //2,
            self.y1 + self.height//2
        )

    def contains_point(self,x:int,y:int) -> bool:
        return (
            self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
        )

class ObjectDetector:
    def __init__(
        self,
        model_path: str = "models/yolo26n_ncnn_model",
        confidence: float = 0.5,
        image_size: int = 320,
        device: str | None = None,
        tracker_config: str = "config/bytetrack_optimize.yaml",
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.image_size = image_size
        self.tracker_config = tracker_config

    def track(self, image:np.ndarray) -> list[Detection]:
        tracking_arguments = {
            "source": image,
            "persist": True,
            "tracker": self.tracker_config,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False
        }

        if self.device is not None:
            tracking_arguments["device"] = self.device

        results = self.model.track(**tracking_arguments)

        tracked_objects: list[Detection] = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            coordinates = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.int().cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist()

            for coordinates, confidence, class_id, track_id in zip(
                coordinates,
                confidences,
                class_ids,
                track_ids
            ):
                x1,y1,x2,y2 = coordinates
                tracked_objects.append(
                    Detection(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=float(confidence),
                        class_id=class_id,
                        label=result.names[class_id],
                        track_id=track_id
                    )
                )
        return tracked_objects
