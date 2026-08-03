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

class ObjectDetector:
    def __init__(self,model_path:str = "yolo26n.pt", confidence: float=0.5,device: str = "cpu") -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def detect(self, image:np.ndarray) -> list[Detection]:
        results = self.model.predict(source=image, conf=self.confidence, device=self.device,verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().tolist()
                    confidence = float(box.conf[0].cpu())
                    class_id = int(box.cls[0].cpu())

                    detections.append(
                        Detection(
                            x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                            confidence=confidence,
                            class_id=class_id,
                            label=self.model.names[class_id]
                        )
                    )

        return detections
