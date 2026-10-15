from threading import Event, Thread
from time import monotonic_ns

import cv2
import numpy as np

from software.camera import Camera, Frame, LatestFrameCamera
from software.config import load_config
from software.detector import Detection
from software.gimbal import SimulatedGimbal
from software.main import TrackingApplication
from software.web import Dashboard
from tools.test_roi_tracker import textured_frame


class PacedCamera(Camera):
    def __init__(self):
        self.stop_event = Event()
        self.sequence = 0

    def start(self):
        self.stop_event.clear()

    def read(self):
        if self.stop_event.wait(0.01):
            return None
        self.sequence += 1
        return Frame(np.full((80, 120, 3), 80, dtype=np.uint8), monotonic_ns(), self.sequence)

    def stop(self):
        self.stop_event.set()


def make_app(factory):
    camera = LatestFrameCamera(PacedCamera())
    dashboard = Dashboard(host='127.0.0.1')
    app = TrackingApplication(load_config(), camera, SimulatedGimbal(), dashboard, factory)
    return app


def frame():
    return Frame(np.zeros((80, 120, 3), dtype=np.uint8), monotonic_ns(), 1)


def test_stop_between_selection_validation_and_resume_wins():
    app = make_app(lambda **kwargs: None)
    selected = Frame(textured_frame(), monotonic_ns(), 1)
    app.dashboard.publish(selected.image, {}, raw_frame=selected)
    app.dashboard._queue({'type': 'roi', 'roi': (20, 20, 40, 40), 'selection_epoch': 0})
    original = app.dashboard.selection_is_current
    def interleaved_stop(command):
        current = original(command)
        assert app.dashboard.app.test_client().post('/api/action', json={'action': 'stop'}).status_code == 200
        return current
    app.dashboard.selection_is_current = interleaved_stop
    app.process_frame(selected)
    app.motors.tick()
    assert not app.motors.get_status().enabled
    assert app.motors.get_status().commanded_pan_speed == 0
    assert not app.manual_mode


def test_new_resume_after_stop_is_not_invalidated_by_queued_stop():
    class Detector:
        def __init__(self, **kwargs):
            pass
        def track(self, image):
            return [Detection(10, 10, 30, 30, .9, 0, 'person', 1)]
    app = make_app(Detector)
    client = app.dashboard.app.test_client()
    client.post('/api/action', json={'action': 'stop'})
    client.post('/api/action', json={'action': 'auto'})
    for _ in range(3):
        app.process_frame(frame())
    app.motors.tick()
    assert app.motors.get_status().enabled
    assert not app.motors._paused


def test_physical_runtime_requires_calibration_confirmation():
    import pytest
    from software.main import build_gimbal
    config = load_config()['gimbal']
    config['backend'] = 'pca9685'
    config['calibration_confirmed'] = False
    with pytest.raises(ValueError, match='calibration is not confirmed'):
        build_gimbal(config)


def test_idle_shows_all_detections_without_motor_movement():
    class Detector:
        def __init__(self, **kwargs):
            pass
        def track(self, image):
            return [Detection(10, 10, 30, 30, .9, 0, 'person', 1),
                    Detection(40, 10, 60, 30, .8, 2, 'car', 2)]
    app = make_app(Detector)
    app.process_frame(frame())
    assert app.detector is not None
    assert app.error is None
    assert app.target is None
    assert len(app.dashboard._annotations) == 2
    app.motors.tick()
    assert not app.motors.get_status().enabled
    assert app.motors.get_status().commanded_pan_speed == 0
    assert app.dashboard.app.test_client().get('/api/status').json['behavior'] == 'IDLE'
    app.queue('next')
    app.process_frame(frame())
    assert [obj.track_id for obj in app.dashboard._annotations] == [1]
    app.queue('stop')
    app.process_frame(frame())
    app.motors.tick()
    assert len(app.dashboard._annotations) == 2
    assert not app.motors.get_status().enabled


def test_idle_detector_failure_is_not_retried_every_frame():
    calls = []
    def broken(**kwargs):
        calls.append(1)
        raise RuntimeError('Model unavailable')
    app = make_app(broken)
    app.process_frame(frame())
    app.process_frame(frame())
    assert len(calls) == 1
    assert app.error is not None
    app.queue('auto')
    app.process_frame(frame())
    assert len(calls) == 2


def test_preview_and_mjpeg_advance_while_inference_is_blocked():
    entered, release = Event(), Event()
    class SlowDetector:
        def __init__(self, **kwargs):
            pass
        def track(self, image):
            entered.set()
            release.wait(2)
            return []
    app = make_app(SlowDetector)
    app.camera.start()
    app.dashboard.start_preview(app.camera)
    app.queue('auto')
    worker = Thread(target=lambda: app.process_frame(frame()))
    worker.start()
    try:
        assert entered.wait(1)
        with app.dashboard._condition:
            assert app.dashboard._condition.wait_for(lambda: app.dashboard._status.get('frame_sequence', 0) >= 3, timeout=1)
            first = app.dashboard._status['frame_sequence']
            assert app.dashboard._condition.wait_for(lambda: app.dashboard._status.get('frame_sequence', 0) > first, timeout=1)
        response = app.dashboard.app.test_client().get('/video_feed', buffered=False)
        chunk = next(response.response)
        assert chunk.startswith(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
        jpeg = chunk.split(b'\r\n\r\n', 1)[1].rstrip(b'\r\n')
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (80, 120, 3)
        response.close()
    finally:
        release.set()
        worker.join(2)
        app.dashboard.stop_preview()
        app.camera.stop()


def test_detector_failure_leaves_preview_available_and_motion_stopped():
    def broken(**kwargs):
        raise RuntimeError('NCNN failed to load')
    app = make_app(broken)
    app.camera.start()
    app.dashboard.start_preview(app.camera)
    try:
        app.queue('auto')
        app.process_frame(frame())
        assert 'NCNN failed' in app.error
        with app.dashboard._condition:
            assert app.dashboard._condition.wait_for(lambda: app.dashboard._jpeg is not None, timeout=1)
        status = app.dashboard.app.test_client().get('/api/status').json
        assert status['camera_state'] == 'running'
        assert not status['gimbal_enabled']
        assert status['behavior'] == 'IDLE'
    finally:
        app.dashboard.stop_preview()
        app.camera.stop()


def test_previous_next_lock_current_detections_and_stop_cancels_cycles():
    class Detector:
        def __init__(self, **kwargs):
            pass
        def track(self, image):
            return [Detection(10, 10, 30, 30, .9, 0, 'person', i) for i in (3, 1, 2)]
    app = make_app(Detector)
    app.queue('previous')
    app.process_frame(frame())
    assert app.selector.selected_track_id == 3
    app.queue('next')
    app.process_frame(frame())
    assert app.selector.selected_track_id == 1
    app.queue('next')
    app.queue('stop')
    app.process_frame(frame())
    assert app.selector.selected_track_id is None
    assert app.behavior.state.name == 'IDLE'


def test_camera_startup_error_is_published_without_crashing():
    stop = Event()
    class BrokenCamera(PacedCamera):
        def start(self):
            stop.set()
            raise RuntimeError('No Pi camera detected')
    app = make_app(lambda **kwargs: None)
    app.camera = LatestFrameCamera(BrokenCamera())
    app.run(stop)
    status = app.dashboard.app.test_client().get('/api/status').json
    assert status['camera_state'] == 'error'
    assert 'No Pi camera' in status['camera_error']
    assert not app.motors.gimbal.get_status().enabled


def test_manual_selection_runs_native_csrt_without_loading_yolo():
    def forbidden(**kwargs):
        raise AssertionError('Manual mode must not load YOLO')
    app = make_app(forbidden)
    image = textured_frame()
    app.dashboard.publish(image, {'camera_state': 'running'})
    client = app.dashboard.app.test_client()
    assert client.post('/api/roi', json={'x': 30, 'y': 30, 'width': 20, 'height': 20}).status_code == 200
    stamp = monotonic_ns()
    for i in range(5):
        app.process_frame(Frame(image.copy(), stamp + i * 30_000_000, i))
        assert app.target is not None
        assert app.roi.state == 'LOCKED'
        assert app.error is None
    assert app.detector is None


def test_roi_rejected_while_camera_is_unavailable():
    app = make_app(lambda **kwargs: None)
    app.dashboard.publish(frame().image, {'camera_state': 'running'})
    app.dashboard.update_status(camera_state='error', camera_error='Camera disconnected')
    response = app.dashboard.app.test_client().post('/api/roi', json={'x': 10, 'y': 10, 'width': 20, 'height': 20})
    assert response.status_code == 409
    assert app.dashboard.pop_commands() == []


def test_camera_cleanup_failure_does_not_abort_retry_or_shutdown():
    class Stop:
        done = False
        waits = []
        def is_set(self): return self.done
        def wait(self, seconds): self.waits.append(seconds)
    stop = Stop()
    class BrokenCamera(PacedCamera):
        starts = 0
        stops = 0
        def start(self):
            self.starts += 1
            if self.starts == 2:
                stop.done = True
            raise RuntimeError('camera disconnected')
        def stop(self):
            self.stops += 1
            raise OSError('camera cleanup failed')
    app = make_app(lambda **kwargs: None)
    source = BrokenCamera()
    app.camera = LatestFrameCamera(source)
    app.run(stop)
    assert source.starts == 2
    assert source.stops == 3  # Two recovery cleanups, then final cleanup.
    assert stop.waits == [2.0, 2.0]
    status = app.dashboard.app.test_client().get('/api/status').json
    assert status['camera_cleanup_error'] == 'camera cleanup failed'
    assert not app.motors.gimbal.get_status().enabled
