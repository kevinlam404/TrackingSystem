"""Regression checks for failures found during the whole-system review."""
from types import SimpleNamespace
from time import monotonic_ns
from threading import Event
import sys

import numpy as np
import pytest

from software.camera import PiCamera
from software.config import load_config, validate_config
from software.controller import OpenLoopTrackingController, TargetError
from software.detector import Detection, ObjectDetector
from software.gimbal import SimulatedGimbal
from software.web import Dashboard
from tools.test_application import make_app, frame


@pytest.mark.parametrize('section,key,value', [
    ('control', 'maximum_pan_speed', float('nan')),
    ('control', 'minimum_pan_speed', 2),
    ('control', 'update_rate_hz', 0),
    ('camera', 'width', 0),
    ('camera', 'fps', True),
    ('gimbal', 'tilt_max_degrees', 180),
])
def test_invalid_settings_rejected(section, key, value):
    config = load_config()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_motion_is_rejected(value):
    gimbal = SimulatedGimbal()
    gimbal.enable()
    with pytest.raises(ValueError):
        gimbal.set_pan_speed(value)
    with pytest.raises(ValueError):
        gimbal.set_tilt_angle(value)
    with pytest.raises(ValueError):
        OpenLoopTrackingController().update(TargetError(value, 0), 0, .025)
    assert gimbal.get_status().commanded_pan_speed == 0


def test_picamera_color_order_and_cleanup_after_stop_failure():
    image = np.array([[[0, 0, 255]]], dtype=np.uint8)
    closed = []
    camera = PiCamera()
    def fail_stop():
        raise OSError('stop failed')
    camera.camera = SimpleNamespace(capture_array=lambda: image,
                                    stop=fail_stop, close=lambda: closed.append(True))
    assert camera.read().image.tolist() == [[[0, 0, 255]]]
    with pytest.raises(OSError):
        camera.stop()
    assert closed == [True] and camera.camera is None


def test_picamera_failed_start_closes_device(monkeypatch):
    closed = []
    def fail_configure(config):
        raise RuntimeError('configuration failed')
    fake = SimpleNamespace(create_video_configuration=lambda **kw: kw,
                           configure=fail_configure, close=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, 'picamera2', SimpleNamespace(Picamera2=lambda: fake))
    camera = PiCamera()
    with pytest.raises(RuntimeError):
        camera.start()
    assert closed == [True] and camera.camera is None


def test_detector_utility_predicts_boxes_without_track_ids():
    class Tensor:
        def __init__(self, value):
            self.value = np.array(value)
        def cpu(self): return self
        def int(self): return self
        def numpy(self): return self.value
        def tolist(self): return self.value.tolist()
    boxes = SimpleNamespace(id=None, xyxy=Tensor([[1, 2, 30, 40]]),
                            conf=Tensor([.9]), cls=Tensor([0]))
    calls = []
    def predict(**kw):
        calls.append(kw)
        return [SimpleNamespace(boxes=boxes, names={0: 'person'})]
    detector = object.__new__(ObjectDetector)
    detector.model = SimpleNamespace(predict=predict)
    detector.confidence, detector.image_size, detector.device = .5, 320, 'cpu'
    found = detector.detect(np.zeros((50, 50, 3), np.uint8))
    assert found[0].label == 'person' and found[0].track_id is None
    assert calls[0]['imgsz'] == 320 and calls[0]['device'] == 'cpu'


def test_local_click_selects_smallest_box_using_shared_runtime():
    class Detector:
        def __init__(self, **kwargs): pass
        def track(self, image):
            return [Detection(5,5,60,60,.9,0,'person',1),
                    Detection(10,10,30,30,.9,1,'object',2)]
    app = make_app(Detector)
    app.select_at_point(20, 20)
    app.process_frame(frame())
    assert app.selector.selected_track_id == 2
    assert [obj.track_id for obj in app.dashboard._annotations] == [2]


def test_stale_selection_rejected_even_without_camera_exception():
    dashboard = Dashboard(host='127.0.0.1')
    dashboard.publish(np.zeros((100,100,3), np.uint8), {'camera_state': 'running'})
    dashboard._published_at_ns = monotonic_ns() - 2_000_000_000
    response = dashboard.app.test_client().post('/api/roi', json=dict(x=10,y=10,width=20,height=20))
    assert response.status_code == 409


def test_preview_reports_frozen_capture(monkeypatch):
    dashboard = Dashboard(host='127.0.0.1')
    timer = iter([0, 2_000_000_000])
    monkeypatch.setattr('software.web.monotonic_ns', lambda: next(timer, 2_000_000_000))
    ready = Event()
    def wait_for_frame(sequence):
        ready.set()
        dashboard._preview_stop.set()
        return None
    dashboard.start_preview(SimpleNamespace(wait_for_frame=wait_for_frame))
    assert ready.wait(1)
    dashboard.stop_preview()
    status = dashboard.app.test_client().get('/api/status').json
    assert status['camera_state'] == 'error'
    assert 'fresh frames' in status['camera_error']


def test_late_snapshot_confirmation_cannot_restart_after_stop():
    from software.camera import Frame
    from tools.test_roi_tracker import textured_frame
    app = make_app(lambda **kwargs: SimpleNamespace(track=lambda image: []))
    raw = Frame(textured_frame(), monotonic_ns(), 1)
    app.dashboard.publish(raw.image, {'camera_state': 'running'}, raw_frame=raw)
    client = app.dashboard.app.test_client()
    token = client.post('/api/snapshot', json={}).json['snapshot_id']
    request = dict(x=30,y=30,width=20,height=20,snapshot_id=token)
    assert client.post('/api/roi', json=request).status_code == 200
    assert client.post('/api/action', json={'action': 'stop'}).status_code == 200
    app.process_frame(raw)
    assert not app.manual_mode and not app.roi.active
    app.motors.tick()
    assert not app.motors.get_status().enabled
    assert client.post('/api/roi', json=request).status_code == 409


def test_web_command_retention_is_bounded():
    dashboard = Dashboard()
    client = dashboard.app.test_client()
    for i in range(100):
        assert client.post('/api/action', json={'action': 'stop'}).status_code == 200
    assert len(dashboard.pop_commands()) == 32


def test_exported_ncnn_resolution_mismatch_is_clear():
    with pytest.raises(ValueError, match='exported'):
        ObjectDetector(model_path='models/yolo26n_ncnn_model', image_size=416)
