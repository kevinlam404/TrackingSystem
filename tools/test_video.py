from __future__ import annotations
import argparse
from time import monotonic
import cv2
from software.camera import OpenCVCamera, PiCamera

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test video capture from camera.")
    parser.add_argument("--source", choices=("webcam", "video", "picamera",), default="webcam")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video", type=str, default = None)
    parser.add_argument("--width", type=int, default = 640)
    parser.add_argument("--height", type=int, default = 480)
    parser.add_argument("--fps", type=int, default = 30)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.source == "picamera":
        camera = PiCamera(width=args.width, height=args.height, fps=args.fps)
    elif args.source == "video":
        if not args.video:
            raise ValueError("Video path must be provided when source is 'video'.")
        camera = OpenCVCamera(source=args.video, width=args.width, height=args.height, fps=args.fps)
    else:
        camera = OpenCVCamera(source=args.device, width=args.width, height=args.height, fps=args.fps)

    frame_count = 0
    measured_fps = 0.0
    measurement_start = monotonic()

    try:
        camera.start()
        while True:
            frame = camera.read()

            if frame is None:
                break
            frame_count += 1
            elapsed_time = monotonic() - measurement_start
            if elapsed_time >= 1.0:
                measured_fps = frame_count / elapsed_time
                frame_count = 0
                measurement_start = monotonic()

            cv2.putText(frame.image, f"FPS: {measured_fps:.1f}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Camera Test", frame.image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"),27):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()