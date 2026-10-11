"""Local selection demo using the same runtime as the browser application."""
from __future__ import annotations

import argparse

from software.config import load_config
from software.main import TrackingApplication, build_camera, build_gimbal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ByteTrack selection with simulated motors")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", choices=("webcam", "picamera"), default="picamera")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--model", default="models/yolo26n_ncnn_model")
    parser.add_argument("--image-size", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["camera"].update(source=args.source, device=args.device, width=args.width,
                            height=args.height, fps=args.fps)
    config["detector"].update(model_path=args.model, confidence=args.confidence,
                              image_size=args.image_size)
    config["gimbal"]["backend"] = "simulated"  # This demo never enables real hardware.
    config["interface"]["local_preview"] = True
    config["interface"]["window_name"] = "ByteTrack Target Selection"
    app = TrackingApplication(config, build_camera(config["camera"]), build_gimbal(config["gimbal"]))
    app.queue("auto")
    try:
        app.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
