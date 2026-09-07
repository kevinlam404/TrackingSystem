from __future__ import annotations
import argparse
from time import monotonic
import cv2
from software.camera import OpenCVCamera, PiCamera, LatestFrameCamera
from software.detector import ObjectDetector, Detection
from software.tracker import TargetSelector
from software.controller import calculate_target_error, OpenLoopTrackingController
from software.behavior import BehaviorState, TrackingBehavior
from software.gimbal import SimulatedGimbal, PCA9685Gimbal
from software.config import load_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ByteTrack target selection"
    )

    parser.add_argument(
        '--config',
        default='config.yaml'
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    camera_config = config['camera']
    detector_config = config['detector']
    behavior_config = config['behavior']
    control_config = config['control']
    gimbal_config = config['gimbal']
    interface_config = config['interface']

    window_name=interface_config['window_name']

    #select camera implementation
    if camera_config['source'] == 'picamera':
        base_camera = PiCamera(
            width=camera_config['width'],
            height=camera_config['height'],
            fps=camera_config['fps'],
        )
    elif camera_config['source'] == 'webcam':
        base_camera = OpenCVCamera(
            source=camera_config['device'],
            width=camera_config['width'],
            height=camera_config['height'],
            fps=camera_config['fps'],
        )
    else:
        raise ValueError(
            "camera.source must be 'picamera' or 'webcam' "
        )

    camera = LatestFrameCamera(base_camera)

    #YOLO performs detection and ByteTrack assigns IDs
    detector =  ObjectDetector(
        model_path=detector_config['model_path'],
        confidence=detector_config['confidence'],
        image_size=detector_config['image_size'],
    )

    #TargetSelector remembers which track ID the user selected
    selector = TargetSelector()
    behavior = TrackingBehavior(
        selector=selector,
        target_label=behavior_config['target_label'],
        acquisition_frames=behavior_config['acquisition_frames'],
        lost_timeout_seconds=behavior_config['lost_timeout_seconds'],
        scan_pan_speed=behavior_config['scan_pan_speed'],
    )
    motion_controller = OpenLoopTrackingController(
        maximum_pan_speed=control_config['maximum_pan_speed'],
        minimum_pan_speed=control_config['minimum_pan_speed'],
        tilt_speed_dps=control_config['tilt_speed_dps'],
        tilt_min_degrees=gimbal_config['tilt_min_degrees'],
        tilt_max_degrees=gimbal_config['tilt_max_degrees'],
    )

    if gimbal_config['backend'] == 'simulated':
        gimbal = SimulatedGimbal(
            tilt_min_degrees=gimbal_config['tilt_min_degrees'],
            tilt_max_degrees=gimbal_config['tilt_max_degrees'],
        )
    elif gimbal_config['backend'] == 'pca9685':
        physical = gimbal_config['pca9685']
        required_pulses = (
            'pan_min_pulse_us',
            'pan_max_pulse_us',
            'tilt_min_pulse_us',
            'tilt_max_pulse_us',
        )
        missing = [key for key in required_pulses if physical[key] is None]
        if missing:
            raise ValueError('PCA9685 calibration values are missing: ' + ", ".join(missing))

        gimbal = PCA9685Gimbal(
            pan_channel=physical['pan_channel'],
            tilt_channel=physical['tilt_channel'],
            pan_min_pulse_us=physical['pan_min_pulse_us'],
            pan_max_pulse_us=physical['pan_max_pulse_us'],
            tilt_min_pulse_us=physical['tilt_min_pulse_us'],
            tilt_max_pulse_us=physical['tilt_max_pulse_us'],
            tilt_min_degrees=gimbal_config['tilt_min_degrees'],
            tilt_max_degrees=gimbal_config['tilt_max_degrees'],
            pan_reversed=physical['pan_reversed'],
            tilt_reversed=physical['tilt_reversed'],
            pan_neutral_trim=physical['pan_neutral_trim'],
            address=physical['address']
        )
    else:
        raise ValueError(
            "gimbal.backend must be " "'simulated' or 'pca9685'"
        )
    


    gimbal.enable()
    last_control_time = monotonic()
    last_valid_frame_time = monotonic()

    latest_objects: list[Detection] =  []
    pending_click: list[tuple[int,int]] = []

    def handle_mouse(event: int, x:int, y:int, flags:int,parameter) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click.clear()
            pending_click.append((x,y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse)

    frame_count = 0
    measured_fps = 0.0
    measurement_start = monotonic()

    try:
        camera.start()

        while True:
            try:
                frame = camera.read()
                if frame is None:
                    if (
                        monotonic() - last_valid_frame_time
                        >= control_config['stale_frame_timeout_seconds']
                    ):
                        if gimbal.get_status().enabled:
                            gimbal.hold()

                    continue
                last_valid_frame_time = monotonic()

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

                    if clicked_target is not None:
                        behavior.lock_manually(clicked_target)

                #look for current selected track ID
                behavior_result = behavior.update(latest_objects)
                selected_target = behavior_result.target

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

                    target_error = calculate_target_error(target=selected_target, frame_width=frame_width, frame_height=frame_height, dead_zone=control_config['dead_zone'])
                
                #gimbal control block
                control_time = monotonic()
                elapsed_seconds = (control_time - last_control_time)
                last_control_time = control_time
                gimbal_status = gimbal.get_status()

                if behavior_result.state == BehaviorState.SCANNING:
                    if not gimbal_status.enabled:
                        gimbal.enable()
                    gimbal.set_pan_speed(behavior_result.requested_scan_pan_speed)
                elif (behavior_result.state == BehaviorState.LOCKED and target_error is not None):
                    if not gimbal_status.enabled:
                        gimbal.enable()
                    
                    command = motion_controller.update(
                        target_error = target_error,
                        current_tilt_degrees=gimbal_status.commanded_tilt_degrees,
                        elapsed_seconds=elapsed_seconds
                    )
                    gimbal.set_pan_speed(command.pan_speed)
                    gimbal.set_tilt_angle(command.tilt_degrees)
                elif behavior_result.state == BehaviorState.IDLE:
                    if gimbal_status.enabled:
                        gimbal.disable()
                else:
                    gimbal.hold()

                    
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
                
                state_name = behavior_result.state.name
                indicator_text = (
                    "LOCK INDICATOR: ON" if behavior_result.lock_indicator_on else "LOCK INDICATOR: OFF"
                )
                indicator_color = (
                    (0,255,0) if behavior_result.lock_indicator_on else (0,0,255)
                )

                cv2.putText(
                    frame.image,
                    f"Behavior: {state_name}",
                    (15,frame_height-75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    frame.image,
                    indicator_text,
                    (15,frame_height-50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    indicator_color,
                    2
                )

                cv2.putText(
                    frame.image,
                    ("Requested scan speed: " f"{behavior_result.requested_scan_pan_speed:+.1f} deg"),
                    (15,frame_height-25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2
                )

                gimbal_status = gimbal.get_status()
                cv2.putText(
                    frame.image,
                    "Gimbal pan command: " f"{gimbal_status.commanded_pan_speed:+.2f}",
                    (15,frame_height-125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0,255,255),
                    2
                )
                cv2.putText(
                    frame.image,
                    "Gimbal tilt command:" f"{gimbal_status.commanded_tilt_degrees:+.1f} deg",
                    (15,frame_height-100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0,255,255),
                    2
                )


                cv2.imshow(window_name, frame.image)
                key = cv2.waitKey(1) & 0xFF
                #Quit
                if key in (ord("q"), 27):
                    break
                #Next Detected Object
                if key == ord("n"):
                    selected = selector.cycle(latest_objects,direction=1)
                    if selected is not None:
                        behavior.lock_manually(selected)

                #Previous Detected Object
                if key == ord("p"):
                    selected = selector.cycle(latest_objects, direction=-1)
                    if selected is not None:
                        behavior.lock_manually(selected)

                #Stop following and resume scanning area
                if key == ord("c"):
                    behavior.start_scanning()

                #completely stop scanning and tracking
                if key == ord("x"):
                    behavior.stop()

                #start scanning again
                if key == ord("s"):
                    behavior.start_scanning()

                #auto select the largetst visible person
                if key == ord("a"):
                    selected = selector.select_largest_person(latest_objects)
                    if selected is not None:
                        behavior.lock_manually(selected)
            except KeyboardInterrupt:
                break
            except Exception:
                if gimbal.get_status().enabled:
                    gimbal.hold()

                raise



    finally:
        camera.stop()
        cv2.destroyAllWindows()
        gimbal.disable()



if __name__ == "__main__":
    main()