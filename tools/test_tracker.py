from __future__ import annotations
import argparse
from time import monotonic
import cv2
from software.camera import OpenCVCamera, PiCamera
from software.detector import ObjectDetector, Detection
from software.tracker import TargetSelector
from software.controller import calculate_target_error

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
    parser.add_argument("--model", default="models/yolo26n_ncnn_model")
    parser.add_argument("--image-size", type=int, default=320)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    #select camera implementation
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

    #YOLO performs detection and ByteTrack assigns IDs
    detector =  ObjectDetector(
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size
    )
    #TargetSelector remembers which track ID the user selected
    selector = TargetSelector()

    latest_objects: list[Detection] =  []
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

            frame_height, frame_width = frame.image.shape[:2]

            #run yolo and bytetrack
            latest_objects = detector.track(frame.image)

            #process a pending mouse click
            if pending_click:
                click_x, click_y = pending_click.pop()

                selector.select_at_point(
                    click_x,
                    click_y,
                    latest_objects
                )

            #look for current selected track ID
            selected_target = selector.get_selected(latest_objects)

            #draw camera center
            frame_center = (frame_width // 2, frame_height // 2)

            cv2.drawMarker(
                frame.image,
                frame_center,
                (255,255,255),
                cv2.MARKER_CROSS,
                24,
                2
            )

            #no selection, show every available box
            if selector.selected_track_id is None:
                objects_to_draw = latest_objects
            #if selected, show only selected box
            elif selected_target is not None:
                objects_to_draw = [selected_target]
            #selected id is temp missing
            else:
                objects_to_draw = []

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

            #calculate control error for the selected target
            target_error = None

            if selected_target is not None:
                center_x, center_y = selected_target.center
                
                cv2.circle(
                    frame.image,
                    (center_x, center_y),
                    6,
                    (0,0,255),
                    -1
                )

                #draw line from camera center to target center
                cv2.line(
                    frame.image,
                    frame_center,
                    (center_x, center_y),
                    (255,0,255),
                    2
                )

                target_error = calculate_target_error(target=selected_target, frame_width=frame_width, frame_height=frame_height, dead_zone=0.05)
                
            #display fps
            frame_count +=1
            elapsed = monotonic() - measurement_start

            if elapsed >= 1.0:
                measured_fps=frame_count/elapsed
                frame_count = 0
                measurement_start = monotonic()

            #general status info
            cv2.putText(
                frame.image,
                f"FPS: {measured_fps:.1f}",
                (15,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2
            )

            cv2.putText(
                frame.image,
                f"Selected ID: {selector.selected_track_id}",
                (15,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2 
            )

            #target error msg when selected target is visible
            if target_error is not None:
                cv2.putText(
                    frame.image,
                    (f"Pan: {target_error.pan:+.2f}"
                     f"{target_error.horizontal_direction}"),
                    (15,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255,255,0),
                    2
                )

                cv2.putText(
                    frame.image,
                    (f"Tilt: {target_error.tilt:+.2f}"
                     f"{target_error.vertical_direction}"),
                    (15,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255,255,0),
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
                selector.select_largest_person(latest_objects)
    finally:
        camera.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()