"""Camera preview, target tracking, and independent motor control."""
from __future__ import annotations

import argparse
import logging
from threading import Event
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
                                control['stale_frame_timeout_seconds'])
        self.selector = TargetSelector()
        self.behavior = TrackingBehavior(selector=self.selector, **config['behavior'])
        self.roi = ROITracker(**config.get('roi_tracking', {}))
        self.detector_factory, self.detector = detector_factory, None
        self.manual_mode = False
        self.error = None
        self.objects = []
        self.target = None
        self._commands = []
        self._fps_start = monotonic()
        self._frames = 0
        self.fps = 0.0
        if dashboard is not None:
            dashboard.motion_action = self.motion_action
            dashboard.motion_status = self.motors.diagnostics

    def motion_action(self, action):
        # HTTP stop does not wait for a frame or inference.
        if action in ('stop', 'estop'):
            self.motors.stop_motion(emergency=action == 'estop')
        elif action == 'reset_estop':
            self.motors.reset_estop()
        elif action == 'center':
            self.motors.center()

    def queue(self, action):
        self.motion_action(action)
        self._commands.append({'type': 'action', 'action': action})

    def _apply_commands(self, frame):
        commands, self._commands = self._commands, []
        if self.dashboard is not None:
            commands += self.dashboard.pop_commands()
        cycles, initial_target = [], None
        for command in commands:
            self.error = None
            if command['type'] == 'roi':
                cycles.clear()
                self.behavior.stop()
                self.roi.clear()
                self.manual_mode = True
                self.motors.resume()
                try:
                    initial_target = self.roi.start(frame.image, command['roi'], frame.timestamp_ns/1e9)
                except (ValueError, RuntimeError, cv2.error) as error:
                    self.error = f'Cannot start object tracking: {error}'
                continue
            action = command['action']
            self.manual_mode, initial_target = False, None
            self.roi.clear()
            if action in ('auto', 'scan', 'clear'):
                cycles.clear()
                self.motors.resume()
                self.behavior.start_scanning()
            elif action in ('stop', 'estop', 'reset_estop', 'center'):
                cycles.clear()
                self.behavior.stop()
                if action != 'center':
                    self.motors.stop_motion()
            elif action in ('previous', 'next'):
                self.motors.resume()
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
                # Idle preview and manual ROI do not load or run YOLO.
                need_detection = self.behavior.state != BehaviorState.IDLE or bool(cycles)
                self.objects = []
                if need_detection:
                    if self.detector is None:
                        self.detector = self.detector_factory(**self.config['detector'])
                    self.objects = self.detector.track(frame.image)
                for direction in cycles:
                    chosen = self.selector.cycle(self.objects, direction)
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
            )
            boxes = [self.target] if self.target is not None else self.objects if self.selector.selected_track_id is None else []
            self.dashboard.annotate(boxes, frame.timestamp_ns)

    def run(self, stop=None):
        stop = Event() if stop is None else stop
        local = self.config['interface'].get('local_preview', False)
        name = self.config['interface'].get('window_name', 'Tracking Gimbal')
        started = False
        self.motors.start()
        if self.dashboard is not None:
            self.dashboard.start_preview(self.camera)
        try:
            while not stop.is_set():
                try:
                    if not started:
                        self.camera.start()
                        started = True
                    frame = self.camera.read()
                    if frame is None:
                        if self.camera.ended:
                            raise RuntimeError('Camera stream ended')
                        continue
                    self.process_frame(frame)
                    if local:
                        image = frame.image.copy()
                        if self.target is not None:
                            obj = self.target
                            cv2.rectangle(image, (obj.x1, obj.y1), (obj.x2, obj.y2), (0, 255, 255), 2)
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
                    self.camera.stop()
                    started = False
                    stop.wait(2.0)
        finally:
            try:
                self.motors.close()
            finally:
                if self.dashboard is not None:
                    self.dashboard.stop_preview()
                self.camera.stop()
                if local:
                    cv2.destroyAllWindows()


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
