from time import monotonic_ns

from software.camera import Frame
from tools.test_application import make_app
from tools.test_roi_tracker import textured_frame


def test_moved_object_is_initialized_from_snapshot_then_confirmed_on_live_frames():
    app = make_app(lambda **kwargs: (_ for _ in ()).throw(AssertionError('Manual selection must not use YOLO')))
    stamp = monotonic_ns()
    old = Frame(textured_frame(30), stamp, 1)
    app.dashboard.publish(old.image, {'camera_state': 'running'}, raw_frame=old)
    client = app.dashboard.app.test_client()
    token = client.post('/api/snapshot', json={}).json['snapshot_id']
    live = Frame(textured_frame(60), stamp+100_000_000, 2)
    app.dashboard.publish(live.image, {'camera_state': 'running'}, raw_frame=live)
    assert client.post('/api/roi', json=dict(x=30,y=30,width=20,height=20,snapshot_id=token)).status_code == 200
    app.process_frame(live)
    assert app.target is None  # Old selected position must never command a motor.
    assert app.roi.state == 'REACQUIRING'
    app.motors.tick()
    assert app.motors.get_status().commanded_pan_speed == 0
    app.process_frame(Frame(textured_frame(60), stamp+200_000_000, 3))
    assert app.target is not None
    assert abs(app.target.center[0]-70) <= 2
    assert app.roi.state == 'LOCKED'
