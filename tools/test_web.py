import numpy as np
import pytest

from software.web import Dashboard


@pytest.fixture
def dashboard():
    return Dashboard(host='127.0.0.1')


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
