from __future__ import annotations
import argparse
from time import monotonic, monotonic_ns
import cv2
from software.camera import OpenCVCamera, PiCamera, LatestFrameCamera
from software.detector import ObjectDetector, Detection
from software.tracker import TargetSelector
from software.controller import calculate_target_error, OpenLoopTrackingController
from software.behavior import BehaviorState, TrackingBehavior
from software.gimbal import SimulatedGimbal, PCA9685Gimbal
from software.config import load_config
from software.roi_tracker import ROITracker
from software.web import Dashboard
from software.motor_loop import MotorLoop



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
    roi_config = config.get('roi_tracking', {})

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

    camera = LatestFrameCamera(base_camera, read_timeout_seconds=0.05)

    roi_tracker = ROITracker(
        velocity_smoothing=roi_config.get('velocity_smoothing', 0.7),
        prediction_frames=roi_config.get('prediction_frames', 1.0),
        prediction_seconds=roi_config.get('prediction_seconds', 0.05),
        reacquire_timeout_seconds=roi_config.get('reacquire_timeout_seconds', 1.0),
        appearance_threshold=roi_config.get('appearance_threshold', 0.8),
    )
    browser_config = interface_config.get('browser', {})
    dashboard = None
    if browser_config.get('enabled', False):
        dashboard = Dashboard(
            host=browser_config.get('host', '0.0.0.0'),
            port=int(browser_config.get('port', 8080)),
        )

    #YOLO performs detection and ByteTrack assigns IDs
    detector =  ObjectDetector(
        model_path=detector_config['model_path'],
        confidence=detector_config['confidence'],
        image_size=detector_config['image_size'],
        tracker_config=detector_config['tracker_config'],
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
            'pan_stop_pulse_us',
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
            pan_stop_pulse_us=physical['pan_stop_pulse_us'],
            pan_max_pulse_us=physical['pan_max_pulse_us'],
            tilt_min_pulse_us=physical['tilt_min_pulse_us'],
            tilt_max_pulse_us=physical['tilt_max_pulse_us'],
            tilt_min_degrees=gimbal_config['tilt_min_degrees'],
            tilt_max_degrees=gimbal_config['tilt_max_degrees'],
            pan_reversed=physical['pan_reversed'],
            tilt_reversed=physical['tilt_reversed'],
            address=physical['address']
        )
    else:
        raise ValueError(
            "gimbal.backend must be " "'simulated' or 'pca9685'"
        )
    


    motors = MotorLoop(gimbal, motion_controller,
                       rate_hz=control_config.get('update_rate_hz', 40),
                       stale_seconds=control_config['stale_frame_timeout_seconds'])
    if dashboard is not None:
        def motion_action(action):
            if action in ('stop', 'estop'):
                motors.stop_motion(emergency=action == 'estop')
            elif action == 'reset_estop':
                motors.reset_estop()
            elif action == 'center':
                motors.center()
        dashboard.motion_action = motion_action
        dashboard.motion_status = motors.diagnostics
    last_valid_frame_time = monotonic()
    latest_frame_timestamp_ns: int | None = None

    latest_objects: list[Detection] =  []
    pending_click: list[tuple[int,int]] = []

    def handle_mouse(event: int, x:int, y:int, flags:int,parameter) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click.clear()
            pending_click.append((x,y))

    frame_count = 0
    measured_fps = 0.0
    measurement_start = monotonic()
    tracking_error = None
    manual_mode = False

    try:
        motors.start()
        camera.start()
        if dashboard is not None:
            dashboard.start()
        if interface_config.get('local_preview', True):
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, handle_mouse)

        while True:
            try:
                frame = camera.read()
                if frame is None:
                    if camera.ended:
                        raise RuntimeError("Camera stream ended")
                    if (
                        monotonic() - last_valid_frame_time
                        >= control_config['stale_frame_timeout_seconds']
                    ):
                        motors.submit('hold', 0)

                    continue
                last_valid_frame_time = monotonic()
                latest_frame_timestamp_ns = frame.timestamp_ns

                frame_height, frame_width = frame.image.shape[:2]

                manual_target = None
                if dashboard is not None:
                    for dashboard_command in dashboard.pop_commands():
                        if dashboard_command['type'] == 'roi':
                            motors.resume()
                            behavior.stop()
                            selector.clear()
                            roi_tracker.clear()
                            manual_mode = True
                            tracking_error = None
                            try:
                                manual_target = roi_tracker.start(
                                    frame.image,
                                    dashboard_command['roi'],
                                    timestamp=frame.timestamp_ns / 1e9,
                                )
                            except (ValueError, RuntimeError, cv2.error) as error:
                                tracking_error = f'Cannot start object tracking: {error}'
                                motors.submit('hold', 0)
                        elif dashboard_command['type'] == 'action':
                            action = dashboard_command['action']
                            tracking_error = None
                            if action != 'center':
                                manual_mode = False
                                manual_target = None
                            if action in ('scan', 'auto', 'clear'):
                                motors.resume()
                                roi_tracker.clear()
                                behavior.start_scanning()
                            elif action in ('stop', 'estop', 'reset_estop'):
                                roi_tracker.clear()
                                behavior.stop()
                                # Emergency latch is applied synchronously by the
                                # HTTP callback; don't replay it after a later reset.
                                motors.stop_motion()
                            elif action == 'center':
                                motors.center()

                #run yolo and bytetrack
                if roi_tracker.active:
                    latest_objects = []
                    if manual_target is None:
                        manual_target = roi_tracker.update(frame.image, timestamp=frame.timestamp_ns / 1e9)
                        if manual_target is None:
                            tracking_error = 'Object temporarily lost; holding motion while searching nearby.' if roi_tracker.active else 'Object lost. Select it again to resume tracking.'
                        else:
                            tracking_error = None
                else:
                    latest_objects = [] if manual_mode else detector.track(frame.image)

                #process a pending mouse click
                if pending_click:
                    click_x, click_y = pending_click.pop()

                    clicked_target = selector.select_at_point(
                        click_x,
                        click_y,
                        latest_objects
                    )

                    if clicked_target is not None:
                        behavior.lock_manually(clicked_target)

                #look for current selected track ID
                behavior_result = behavior.update(latest_objects)
                selected_target = manual_target or behavior_result.target

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
                if manual_target is not None:
                    objects_to_draw = [manual_target]
                elif selector.selected_track_id is None:
                    objects_to_draw = latest_objects
                #if selected, show only selected box
                elif selected_target is not None:
                    objects_to_draw = [selected_target]
                #selected id is temp missing
                else:
                    objects_to_draw = []

                #Drawing loop
                for tracked_object in objects_to_draw:
                    is_selected = tracked_object is manual_target or (
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
                frame_age_seconds = (
                    monotonic_ns() - latest_frame_timestamp_ns
                ) / 1_000_000_000
                observed_at = latest_frame_timestamp_ns / 1e9
                if manual_mode:
                    motors.submit('track' if manual_target is not None else 'hold', observed_at, target_error)
                elif behavior_result.state == BehaviorState.SCANNING:
                    motors.submit('scan', observed_at, scan_speed=behavior_result.requested_scan_pan_speed)
                elif (behavior_result.state == BehaviorState.LOCKED and target_error is not None):
                    motors.submit('track', observed_at, target_error)
                elif behavior_result.state == BehaviorState.IDLE:
                    motors.submit('idle', observed_at)
                else:
                    motors.submit('hold', observed_at)

                    
                #display fps
                # Keep browser video free of the local diagnostic text overlays.
                browser_image = frame.image.copy() if dashboard is not None else None
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

                gimbal_status = motors.get_status()
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


                if dashboard is not None:
                    status = motors.get_status()
                    dashboard.publish(
                        browser_image,
                        {
                            'mode': 'manual_roi' if manual_mode else 'auto_person',
                            'behavior': roi_tracker.state if manual_mode else behavior_result.state.name,
                            'motor_stale': motors.stale,
                            'motor_ticks': motors.ticks,
                            'tracking_error': tracking_error,
                            'frame_sequence': frame.sequence,
                            'selected_id': selector.selected_track_id,
                            'fps': round(measured_fps, 1),
                            'frame_age_ms': round(frame_age_seconds * 1000.0, 1),
                            'gimbal_enabled': status.enabled,
                            'pan_speed': round(status.commanded_pan_speed, 3),
                            'tilt_degrees': round(status.commanded_tilt_degrees, 2),
                            'fault': motors.fault or status.fault,
                        },
                    )

                key = 255
                if interface_config.get('local_preview', True):
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
                    manual_mode = False
                    motors.resume()
                    roi_tracker.clear()
                    behavior.start_scanning()

                #completely stop scanning and tracking
                if key == ord("x"):
                    manual_mode = False
                    motors.stop_motion()
                    roi_tracker.clear()
                    behavior.stop()

                #start scanning again
                if key == ord("s"):
                    manual_mode = False
                    motors.resume()
                    roi_tracker.clear()
                    behavior.start_scanning()

                #auto select the largetst visible person
                if key == ord("a"):
                    manual_mode = False
                    motors.resume()
                    roi_tracker.clear()
                    selected = selector.select_largest_person(latest_objects)
                    if selected is not None:
                        behavior.lock_manually(selected)
            except KeyboardInterrupt:
                break
            except Exception:
                motors.stop_motion()

                raise



    finally:
        try:
            motors.close()
        finally:
            try:
                camera.stop()
            finally:
                cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
