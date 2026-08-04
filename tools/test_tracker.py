from __future__ import annotations
import argparse
from time import monotonic
import cv2
from software.camera import OpenCVCamera, PiCamera
from software.detector import ObjectDetector, Detection
from software.tracker import TargetSelector

WINDOW_NAME = "ByteTrack Target Selection"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ByteTrack target selection"
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

    selector = TargetSelector()
    latest_object: list[Detection] =  []
    pending_click: list[tuple[int,int]] = []

    def handle_mouse(event: int, x:int, y:int, flags:int,parameter) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click.clear()
            pending_click.append((x,y))

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    frame_count = 0
    measured_fps = 0.0
    measurement_start = monotonic()

    try:
        camera.start()

        while True:
            frame = camera.read()
            if frame is None:
                break

            latest_objects = detector.track(frame.image)
            if pending_click:
                click_x, click_y = pending_click.pop()

                selector.select_at_point(
                    click_x,
                    click_y,
                    latest_objects
                )


            selected_target = selector.get_selected(latest_objects)

            if selector.selected_track_id is None:
                #Selection mode: display every available target
                objects_to_draw = latest_objects
            else:
                #Tracking mode: display only the selected target
                objects_to_draw = (
                    [selected_target]
                    if selected_target is not None else []
                )

            #Drawing loop
            for tracked_object in objects_to_draw:
                is_selected = (
                    tracked_object.track_id == selector.selected_track_id
                )

                if is_selected:
                    color = (0,255,255)
                    thickness = 4
                else:
                    color = (0,255,0)
                    thickness = 2

                cv2.rectangle(
                    frame.image,
                    (tracked_object.x1, tracked_object.y1),
                    (tracked_object.x2, tracked_object.y2),
                    color,
                    thickness
                )

                label=(
                    f"{tracked_object.label}"
                    f"ID:{tracked_object.track_id}"
                    f"{tracked_object.confidence:.2f}"
                )

                cv2.putText(
                    frame.image,
                    label,
                    (tracked_object.x1, max(20,tracked_object.y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2
                )

            #lost target msg
            if(selector.selected_track_id is not None and selected_target is None):
                cv2.putText(
                    frame.image,
                    "SELECTED TARGET LOST - Press C to clear",
                    (15,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0,0,255),
                    2
                )
            
            if selected_target is not None:
                center_x, center_y = selected_target.center
                cv2.circle(
                    frame.image,
                    (center_x, center_y),
                    6,
                    (0,0,255),
                    -1
                )

            frame_count +=1
            elapsed = monotonic() - measurement_start

            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                measurement_start = monotonic()

            cv2.putText(
                frame.image,
                f"FPS: {measured_fps:.1f}",
                (15,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2
            )

            cv2.imshow(WINDOW_NAME, frame.image)
            key = cv2.waitKey(1) & 0xFF
            #Quit
            if key in (ord("q"), 27):
                break
            #Next Detected Object
            if key == ord("n"):
                selector.cycle(latest_objects,direction=1)
            #Previous Detected Object
            if key == ord("p"):
                selector.cycle(latest_objects, direction=-1)
            #Stop following selected target
            if key == ord("c"):
                selector.clear()
            #auto select the largetst visible person
            if key == ord("a"):
                selector.select_largest_person(latest_object)
    finally:
        camera.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()