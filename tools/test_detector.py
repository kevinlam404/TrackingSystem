import argparse
from time import monotonic
import cv2
from software.camera import OpenCVCamera, PiCamera
from software.detector import ObjectDetector

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test live object detection from camera.")
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default = 640)
    parser.add_argument("--height", type=int, default = 480)
    parser.add_argument("--fps", type=int, default = 30)
    parser.add_argument("--confidence", type=float,default=0.5)
    parser.add_argument("--model", default="yolo26n.pt")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    camera = OpenCVCamera(
        source=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

    detector = ObjectDetector(
        model_path=args.model,
        confidence=args.confidence,
        device="cpu"
    )

    measured_fps = 0.0
    frame_count = 0
    mesurement_start = monotonic()

    try:
        camera.start()
        while True:
            frame = camera.read()

            if frame is None:
                break

            detections = detector.detect(frame.image)

            for detection in detections:
                cv2.rectangle(
                    frame.image,
                    (detection.x1, detection.y1),
                    (detection.x2, detection.y2),
                    (0, 255, 0),
                    2
                )

                text = f"{detection.label} {detection.confidence:.2f}"
                cv2.putText(
                    frame.image,
                    text,
                    (detection.x1,max(20,detection.y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            frame_count += 1
            elapsed_time = monotonic() - mesurement_start

            if elapsed_time >= 1.0:
                measured_fps = frame_count / elapsed_time
                frame_count = 0
                mesurement_start = monotonic()

            cv2.putText(frame.image, f"FPS: {measured_fps:.1f}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Object Detection", frame.image)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"),27):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()