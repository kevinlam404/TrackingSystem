"""Camera preview, target tracking, and independent motor control."""
from __future__ import annotations

import argparse
import logging
from threading import Event
from queue import SimpleQueue, Empty
from time import monotonic, monotonic_ns

import cv2

from software.behavior import BehaviorState, TrackingBehavior
from software.camera import LatestFrameCamera, OpenCVCamera, PiCamera
from software.config import load_config
from software.controller import OpenLoopTrackingController, calculate_target_error
from software.detector import ObjectDetector
from software.gimbal import PCA9685Gimbal, SimulatedGimbal
from software.motor_loop import MotorLoop
from software.roi_tracker import ROITracker
from software.tracker import TargetSelector
from software.web import Dashboard

LOG = logging.getLogger(__name__)


def build_camera(config):
    options = {key: config[key] for key in ('width', 'height', 'fps')}
    if config['source'] == 'picamera':
        source = PiCamera(**options)
    elif config['source'] == 'webcam':
        source = OpenCVCamera(source=config['device'], **options)
    else:
        raise ValueError("camera.source must be 'picamera' or 'webcam'")
    return LatestFrameCamera(source)


def build_gimbal(config):
    limits = {key: config[key] for key in ('tilt_min_degrees', 'tilt_max_degrees')}
    if config['backend'] == 'simulated':
        return SimulatedGimbal(**limits)
    if config['backend'] != 'pca9685':
        raise ValueError("gimbal.backend must be 'simulated' or 'pca9685'")
    physical = config['pca9685']
    if not config.get('calibration_confirmed', False):
        raise ValueError('Physical servo calibration is not confirmed. Run the guarded bench test, verify neutral/directions/limits and power cutoff, then set gimbal.calibration_confirmed: true.')
    required = ('pan_min_pulse_us', 'pan_stop_pulse_us', 'pan_max_pulse_us',
                'tilt_min_pulse_us', 'tilt_max_pulse_us')
    missing = [key for key in required if physical.get(key) is None]
    if missing:
        raise ValueError('Missing servo calibration: ' + ', '.join(missing))
    return PCA9685Gimbal(**limits, **physical)


class TrackingApplication:
    """Testable vision loop; browser frames are delivered independently."""

    def __init__(self, config, camera, gimbal, dashboard=None, detector_factory=ObjectDetector):
        self.config, self.camera, self.dashboard = config, camera, dashboard
        control, limits = config['control'], config['gimbal']
        controller = OpenLoopTrackingController(
            maximum_pan_speed=control['maximum_pan_speed'],
            minimum_pan_speed=control['minimum_pan_speed'],
            tilt_speed_dps=control['tilt_speed_dps'],
            tilt_min_degrees=limits['tilt_min_degrees'],
            tilt_max_degrees=limits['tilt_max_degrees'],
        )
        self.motors = MotorLoop(gimbal, controller, control.get('update_rate_hz', 40),
                                control['stale_frame_timeout_seconds'],
                                pan_acceleration=control.get('pan_acceleration_per_second', 1.0))
        self.selector = TargetSelector()
        self.behavior = TrackingBehavior(selector=self.selector, **config['behavior'])
        self.roi = ROITracker(**config.get('roi_tracking', {}))
        self.detector_factory, self.detector = detector_factory, None
        self._detector_failed = False
        self.manual_mode = False
        self.error = None
        self.objects = []
        self.target = None
        self._commands = SimpleQueue()
        self._fps_start = monotonic()
        self._frames = 0
        self.fps = 0.0
        if dashboard is not None:
            dashboard.motion_action = self.motion_action
            dashboard.motion_status = self.motors.diagnostics
            dashboard.motion_generation = self.motors.generation
            dashboard.annotation_max_age_seconds = control['stale_frame_timeout_seconds']

    def motion_action(self, action):
        # HTTP stop does not wait for a frame or inference.
        if action in ('stop', 'estop'):
            self.motors.stop_motion(emergency=action == 'estop')
        elif action == 'reset_estop':
            self.motors.reset_estop()
        elif action == 'center':
            self.motors.center()

    def queue(self, action):
        generation = self.motors.generation()
        self.motion_action(action)
        self._commands.put({'type': 'action', 'action': action, 'motion_generation': generation})

    def select_at_point(self, x, y):
        self._commands.put({'type': 'point', 'point': (x, y),
                            'motion_generation': self.motors.generation()})

    def _apply_commands(self, frame):
        batch_generation = self.motors.generation()
        commands = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except Empty:
                break
        if self.dashboard is not None:
            commands += self.dashboard.pop_commands()
        cycles, initial_target = [], None
        for command in commands:
            resumes = command['type'] in ('point', 'roi') or command.get('action') in ('auto', 'scan', 'clear', 'previous', 'next')
            if command['type'] == 'roi' and self.dashboard is not None and not self.dashboard.selection_is_current(command):
                continue
            if resumes and not self.motors.resume(command.get('motion_generation', batch_generation)):
                continue
            self.error = None
            if command['type'] == 'point':
                self.manual_mode, initial_target = False, None
                self.roi.clear()
                self._detector_failed = False
                cycles.append(command['point'])
                continue
            if command['type'] == 'roi':
                if self.dashboard is not None and not self.dashboard.selection_is_current(command):
                    continue
                initial_target = None
                cycles.clear()
                self.behavior.stop()
                self.roi.clear()
                self.manual_mode = True
                try:
                    selected_frame = command.get('frame', frame)
                    initial_target = self.roi.start(selected_frame.image, command['roi'], selected_frame.timestamp_ns/1e9)
                    if selected_frame.sequence != frame.sequence or selected_frame.timestamp_ns != frame.timestamp_ns:
                        # Do not command motors using a box on an old snapshot.
                        self.roi.begin_recovery(frame.timestamp_ns/1e9)
                        initial_target = None
                except (ValueError, RuntimeError, cv2.error) as error:
                    self.motors.stop_motion()
                    self.error = f'Cannot start object tracking: {error}'
                continue
            action = command['action']
            self.manual_mode, initial_target = False, None
            self.roi.clear()
            if action in ('auto', 'scan', 'clear'):
                self._detector_failed = False
                cycles.clear()
                self.behavior.start_scanning()
            elif action in ('stop', 'estop', 'reset_estop', 'center'):
                cycles.clear()
                self.behavior.stop()
                if action != 'center' and 'motion_generation' not in command:
                    self.motors.stop_motion()
            elif action in ('previous', 'next'):
                self._detector_failed = False
                cycles.append(-1 if action == 'previous' else 1)
        return initial_target, cycles

    def process_frame(self, frame):
        initial_target, cycles = self._apply_commands(frame)
        try:
            if self.manual_mode:
                self.objects, self.target = [], initial_target
                if self.target is None and self.roi.active:
                    self.target = self.roi.update(frame.image, frame.timestamp_ns/1e9)
                if self.target is not None:
                    self.error = None
                elif self.error is None:
                    self.error = ('Object temporarily lost; holding while searching nearby.'
                                  if self.roi.active else 'Object lost. Select it again to resume.')
            else:
                # Browse every YOLO/ByteTrack box in idle, without moving motors.
                # Manual ROI remains independent of YOLO. Do not retry a broken
                # model every frame; an explicit detection action retries it.
                self.objects = []
                if not self._detector_failed:
                    if self.detector is None:
                        self.detector = self.detector_factory(**self.config['detector'])
                    self.objects = self.detector.track(frame.image)
                for direction in cycles:
                    chosen = (self.selector.select_at_point(*direction, self.objects)
                              if isinstance(direction, tuple) else self.selector.cycle(self.objects, direction))
                    if chosen is None:
                        self.behavior.stop()
                        self.error = 'No detected targets available.'
                    else:
                        self.behavior.lock_manually(chosen)
            result = self.behavior.update(self.objects)
            if not self.manual_mode:
                self.target = result.target
            error = None
            if self.target is not None:
                height, width = frame.image.shape[:2]
                error = calculate_target_error(self.target, width, height,
                                               self.config['control']['dead_zone'])
            mode = 'hold'
            if self.target is not None:
                mode = 'track'
            elif not self.manual_mode:
                if result.state == BehaviorState.SCANNING:
                    mode = 'scan'
                elif result.state == BehaviorState.IDLE:
                    mode = 'idle'
            self.motors.submit(mode, frame.timestamp_ns/1e9, error, result.requested_scan_pan_speed)
        except Exception as error:
            LOG.exception('Vision failed; preview remains available')
            self.error = f'Vision error: {error}'
            self.target, self.objects = None, []
            self.behavior.stop()
            self.roi.clear()
            self.detector = None
            self._detector_failed = not self.manual_mode
            self.motors.stop_motion()
        self._frames += 1
        elapsed = monotonic() - self._fps_start
        if elapsed >= 1:
            self.fps = round(self._frames / elapsed, 1)
            self._frames, self._fps_start = 0, monotonic()
        if self.dashboard is not None:
            self.dashboard.update_status(
                mode='manual_roi' if self.manual_mode else 'auto_person',
                behavior=self.roi.state if self.manual_mode else self.behavior.state.name,
                selected_id=self.selector.selected_track_id, tracking_error=self.error,
                fps=self.fps, frame_age_ms=round((monotonic_ns()-frame.timestamp_ns)/1e6, 1),
                vision_frame_age_ms=round((monotonic_ns()-frame.timestamp_ns)/1e6, 1),
                servo_backend=self.config['gimbal']['backend'],
                servo_position_feedback=False, sensor_latency_measured=False,
            )
            boxes = [self.target] if self.target is not None else self.objects if self.selector.selected_track_id is None else []
            self.dashboard.annotate(boxes, frame.timestamp_ns,
                                    selected_id=self.target.track_id if self.target is not None else None)

    def run(self, stop=None):
        stop = Event() if stop is None else stop
        local = self.config['interface'].get('local_preview', False)
        name = self.config['interface'].get('window_name', 'Tracking Gimbal')
        started = False
        last_frame_at = monotonic()
        if local:
            cv2.namedWindow(name)
            cv2.setMouseCallback(name, lambda event, x, y, flags, param:
                                self.select_at_point(x, y) if event == cv2.EVENT_LBUTTONDOWN else None)
        self.motors.start()
        if self.dashboard is not None:
            self.dashboard.start_preview(self.camera)
        try:
            while not stop.is_set():
                try:
                    if not started:
                        self.camera.start()
                        started = True
                        last_frame_at = monotonic()
                    frame = self.camera.read()
                    if frame is None:
                        if self.camera.ended:
                            raise RuntimeError('Camera stream ended')
                        if monotonic() - last_frame_at > 2.0:
                            raise RuntimeError('Camera is not producing fresh frames')
                        continue
                    last_frame_at = monotonic()
                    self.process_frame(frame)
                    if local:
                        image = frame.image.copy()
                        boxes = [self.target] if self.target is not None else self.objects if self.selector.selected_track_id is None else []
                        for obj in boxes:
                            color = (255, 255, 0) if self.target is not None else (0, 220, 0)
                            cv2.rectangle(image, (obj.x1, obj.y1), (obj.x2, obj.y2), color, 3 if self.target is not None else 2)
                            cv2.putText(image, f'{obj.label} ID {obj.track_id}', (obj.x1, max(18, obj.y1-6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        height, width = image.shape[:2]
                        center = (width//2, height//2)
                        cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
                        if self.target is not None:
                            cv2.circle(image, self.target.center, 5, (0, 0, 255), -1)
                            cv2.line(image, center, self.target.center, (255, 0, 255), 2)
                        cv2.putText(image, f'{self.behavior.state.name} | {self.fps:.1f} FPS', (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.imshow(name, image)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q')):
                            break
                        action = {ord('n'): 'next', ord('p'): 'previous', ord('a'): 'auto',
                                  ord('s'): 'scan', ord('c'): 'clear', ord('x'): 'stop'}.get(key)
                        if action:
                            self.queue(action)
                except (RuntimeError, OSError) as error:
                    LOG.exception('Camera unavailable; retrying in two seconds')
                    self.motors.stop_motion()
                    self.behavior.stop()
                    self.roi.clear()
                    if self.dashboard is not None:
                        self.dashboard.update_status(camera_state='error', camera_error=str(error))
                    self._stop_camera()
                    started = False
                    stop.wait(2.0)
        finally:
            try:
                self.motors.close()
            finally:
                if self.dashboard is not None:
                    self.dashboard.stop_preview()
                self._stop_camera()
                if local:
                    cv2.destroyAllWindows()

    def _stop_camera(self):
        """Cleanup errors must not kill recovery or bypass final shutdown."""
        try:
            self.camera.stop()
        except Exception as error:
            LOG.exception('Camera cleanup failed; motion remains stopped')
            if self.dashboard is not None:
                self.dashboard.update_status(camera_cleanup_error=str(error))


def main():
    parser = argparse.ArgumentParser(description='Run tracking gimbal and browser dashboard')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    browser = config['interface'].get('browser', {})
    dashboard = Dashboard(host=browser.get('host', '127.0.0.1'), port=int(browser.get('port', 8080))) if browser.get('enabled', False) else None
    if dashboard is not None:
        dashboard.update_status(camera_state='starting', behavior='IDLE')
        dashboard.start()
    try:
        app = TrackingApplication(config, build_camera(config['camera']), build_gimbal(config['gimbal']), dashboard)
        app.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
