from __future__ import annotations
import argparse
from time import monotonic
import cv2
from software.camera import OpenCVCamera, PiCamera
from software.detector import ObjectDetector
from software.tracker import ObjectTracker, TrackingState

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test person detection and tracking"
    )

    parser.add_argument(
        "--source",
        choices=("webcam", "picamera"),
        default="picamera"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--model", default="yolo26n.pt")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.source == "picamera":
        camera = PiCamera(
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
    else:
        camera = OpenCVCamera(
            source=args.device,
            width=args.width,
            height=args.height,
            fps=args.fps
        )

    detector =  ObjectDetector(
        model_path=args.model,
        confidence=args.confidence,
        device="cpu"
    )

    tracker = ObjectTracker()
    tracker.search()

    frame_count = 0
    measured_fps = 0.0
    measurement_start = monotonic()

    try:
        camera.start()
        while True:
            frame = camera.read()

            if frame is None:
                break

            target = None

            #Run YOLO only when we need to find a target
            if tracker.state in (TrackingState.SEARCHING, TrackingState.LOST):
                detections = detector.detect(frame.image)
                detection = tracker.select_largest_person(
                    detections
                )
                if detection is not None:
                    target = tracker.start(
                        frame.image,
                        detection
                    )

            #Once a target is acquired, update it with KCF
            elif tracker.state == TrackingState.TRACKING:
                target = tracker.update(frame.image)

            #Draw the tracked target
            if target is not None:
                cv2.rectangle(
                    frame.image,
                    (target.x1, target.y1),
                    (target.x2, target.y2),
                    (0,255,255),
                    2,
                )
                center_x, center_y = target.center

                cv2.circle(
                    frame.image,
                    (center_x, center_y),
                    5,
                    (0,0,255),
                    -1
                )

                cv2.putText(
                    frame.image,
                    f"Tracking: {target.label}",
                    (target.x1, max(20, target.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255),
                    2
                )

            frame_count +=1
            elapsed = monotonic() - measurement_start
            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                measurement_start = monotonic()
            status_text = tracker.state.name
            cv2.putText(
                frame.image,
                f"State: {status_text}",
                (15,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2,
            )

            cv2.putText(
                frame.image,
                f"FPS: {measured_fps:.1f}",
                (15,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2,
            )

            cv2.imshow(
                "Person Tracking Test",
                frame.image
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if key == ord("r"):
                tracker.search()

    finally:
        tracker.reset()
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()