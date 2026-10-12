import numpy as np
import pytest

from software.web import Dashboard


@pytest.fixture
def dashboard():
    return Dashboard(host='127.0.0.1')


@pytest.mark.parametrize('age,visible', [(0.35, True), (0.6, False)])
def test_overlay_retained_until_control_freshness_limit(dashboard, age, visible):
    from threading import Event
    from time import monotonic_ns
    from software.camera import Frame
    from software.detector import Detection
    now = monotonic_ns()
    dashboard.annotate([Detection(10, 10, 30, 30, .9, 0, 'person', 1)], now-int(age*1e9), 1)
    published = Event()
    images = []
    class Camera:
        def wait_for_frame(self, sequence):
            return Frame(np.zeros((80, 100, 3), dtype=np.uint8), now, 1)
    def publish(image, status, raw_frame=None):
        images.append(image)
        dashboard._preview_stop.set()
        published.set()
    dashboard.publish = publish
    dashboard.start_preview(Camera())
    try:
        assert published.wait(1)
    finally:
        dashboard.stop_preview()
    assert bool(np.any(images[0])) == visible


def test_dashboard_renders_selection_controls(dashboard):
    response = dashboard.app.test_client().get('/')
    assert response.status_code == 200
    for text in (b'Track selection', b'Redraw', b'Cancel', b'Select object'):
        assert text in response.data


def test_roi_requires_camera_frame(dashboard):
    response = dashboard.app.test_client().post('/api/roi', json=dict(x=0, y=0, width=20, height=20))
    assert response.status_code == 409
    assert dashboard.pop_commands() == []


@pytest.mark.parametrize('roi', [(-1, 0, 20, 20), (0, 0, 7, 20), (90, 0, 20, 20), (0, 90, 20, 20)])
def test_invalid_roi_never_queues(dashboard, roi):
    dashboard.publish(np.zeros((100, 100, 3), dtype=np.uint8), {})
    response = dashboard.app.test_client().post('/api/roi', json=dict(zip(('x', 'y', 'width', 'height'), roi)))
    assert response.status_code == 400
    assert dashboard.pop_commands() == []


def test_valid_roi_and_stop_queue_in_order(dashboard):
    client = dashboard.app.test_client()
    dashboard.publish(np.zeros((100, 100, 3), dtype=np.uint8), {'frame_sequence': 1})
    assert client.post('/api/action', json={'action': 'stop'}).status_code == 200
    assert client.post('/api/roi', json=dict(x=10, y=20, width=30, height=40)).status_code == 200
    assert dashboard.pop_commands() == [dict(type='action', action='stop'), dict(type='roi', roi=(10, 20, 30, 40))]
    assert client.get('/api/status').json['frame_sequence'] == 1


def test_stop_bypasses_frame_queue_and_status_reads_live_motor_state(dashboard):
    actions = []
    dashboard.motion_action = actions.append
    dashboard.motion_status = lambda: {'emergency_stopped': True, 'gimbal_enabled': False}
    client = dashboard.app.test_client()
    assert client.post('/api/action', json={'action': 'estop'}).status_code == 200
    assert actions == ['estop']
    assert client.get('/api/status').json['emergency_stopped'] is True


@pytest.mark.parametrize('action', ['previous', 'next'])
def test_browser_target_navigation(dashboard, action):
    client = dashboard.app.test_client()
    assert client.post('/api/action', json={'action': action}).status_code == 200
    assert dashboard.pop_commands() == [{'type': 'action', 'action': action}]
    page = client.get('/').data
    assert b'Previous target' in page and b'Next target' in page


def test_snapshot_retains_exact_raw_frame_and_is_single_use(dashboard):
    from time import monotonic_ns
    from software.camera import Frame
    raw = Frame(np.full((100, 100, 3), 60, dtype=np.uint8), monotonic_ns(), 10)
    dashboard.publish(raw.image, {'camera_state': 'running'}, raw_frame=raw)
    client = dashboard.app.test_client()
    snapshot = client.post('/api/snapshot', json={})
    assert snapshot.status_code == 200
    token = snapshot.json['snapshot_id']
    assert snapshot.json['image'].startswith('data:image/jpeg;base64,')
    newer = Frame(np.full((100, 100, 3), 90, dtype=np.uint8), monotonic_ns(), 11)
    dashboard.publish(newer.image, {'camera_state': 'running'}, raw_frame=newer)
    request = dict(x=10, y=10, width=20, height=20, snapshot_id=token)
    assert client.post('/api/roi', json=request).status_code == 200
    command = dashboard.pop_commands()[-1]
    assert command['frame'] is raw
    assert client.post('/api/roi', json=request).status_code == 409


@pytest.mark.parametrize('endpoint', ['/api/action', '/api/roi'])
def test_non_object_json_is_rejected_without_server_error(dashboard, endpoint):
    assert dashboard.app.test_client().post(endpoint, json=['invalid']).status_code == 400


def test_expired_snapshot_is_rejected(dashboard):
    from time import monotonic_ns
    from software.camera import Frame
    raw = Frame(np.zeros((100,100,3), dtype=np.uint8), monotonic_ns(), 1)
    dashboard.publish(raw.image, {'camera_state': 'running'}, raw_frame=raw)
    client = dashboard.app.test_client()
    token = client.post('/api/snapshot', json={}).json['snapshot_id']
    dashboard._snapshots[token] = (monotonic_ns()-31_000_000_000, raw)
    assert client.post('/api/roi', json=dict(x=0,y=0,width=20,height=20,snapshot_id=token)).status_code == 409


def test_jpeg_fallback_is_decodable_and_uncached(dashboard):
    dashboard.publish(np.full((80,120,3), 100, dtype=np.uint8), {})
    response = dashboard.app.test_client().get('/frame.jpg')
    assert response.status_code == 200
    assert response.mimetype == 'image/jpeg'
    assert response.headers['Cache-Control'] == 'no-store'
    import cv2
    image = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (80,120,3)


def test_jpeg_fallback_reports_missing_and_stale_frames(dashboard):
    client = dashboard.app.test_client()
    assert client.get('/frame.jpg').status_code == 503
    dashboard.publish(np.zeros((80,120,3), dtype=np.uint8), {})
    dashboard._published_at_ns = 0
    assert client.get('/frame.jpg').status_code == 503


def test_browser_uses_timed_jpeg_transport(dashboard):
    page = dashboard.app.test_client().get('/').data
    assert b"fetch('/frame.jpg" in page
    assert b'src="/video_feed"' not in page
    assert b'AbortSignal.timeout' not in page
    assert b'Live video connected.' in page
    assert b'previewPeriodMs=1000/30' in page
    assert b'previewPeriodMs-(performance.now()-requestStartedAt)' in page
    assert b'let retryDelay=100;' not in page
    assert b'X-Frame-Sequence' in page


def test_preview_sequence_header_identifies_duplicate_frames(dashboard):
    dashboard.publish(np.zeros((80, 120, 3), dtype=np.uint8), {'frame_sequence': 42})
    client = dashboard.app.test_client()
    assert client.get('/frame.jpg').headers['X-Frame-Sequence'] == '42'
    assert client.get('/frame.jpg').headers['X-Frame-Sequence'] == '42'
    dashboard.publish(np.zeros((80, 120, 3), dtype=np.uint8), {'frame_sequence': 43})
    assert client.get('/frame.jpg').headers['X-Frame-Sequence'] == '43'


def test_json_deadline_covers_body_consumption(dashboard):
    page = dashboard.app.test_client().get('/').data
    helper = page.split(b'async function requestJSON', 1)[1].split(b'function notice', 1)[0]
    assert helper.index(b'await response.json()') < helper.index(b'clearTimeout(timeout)')
    assert b"requestJSON('/api/status'" in page


def test_selected_annotations_are_cyan_and_detections_green(dashboard):
    from threading import Event
    from time import monotonic_ns
    import cv2
    from software.camera import Frame
    from software.detector import Detection
    ready = Event()
    stamp = monotonic_ns()
    objects = [Detection(10,30,30,50,.9,0,'person',1),
               Detection(50,30,70,50,.9,1,'object',2)]
    dashboard.annotate(objects, stamp, selected_id=2)
    def wait_for_frame(sequence):
        if sequence == -1:
            return Frame(np.zeros((80,100,3),np.uint8), stamp, 1)
        ready.set()
        dashboard._preview_stop.set()
        return None
    from types import SimpleNamespace
    dashboard.start_preview(SimpleNamespace(wait_for_frame=wait_for_frame))
    assert ready.wait(1)
    dashboard.stop_preview()
    image = cv2.imdecode(np.frombuffer(dashboard._jpeg,np.uint8),cv2.IMREAD_COLOR)
    b,g,r = image[40,10].astype(int)
    assert g > b+80 and g > r+80
    b,g,r = image[40,50].astype(int)
    assert b > r+80 and g > r+80
