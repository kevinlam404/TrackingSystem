from software.behavior import BehaviorState, TrackingBehavior
from software.detector import Detection
from software.tracker import TargetSelector

def make_person(track_id: int = 1, size: int = 100) -> Detection:
    return Detection(
        x1=100,
        y1=100,
        x2=100+size,
        y2=100+size,
        confidence=0.9,
        class_id=0,
        label='person',
        track_id=track_id
    )

def make_behavior(acquisition_frames: int=3, lost_timeout_seconds: float=1.0) -> TrackingBehavior:
    selector = TargetSelector()
    return TrackingBehavior(
        selector=selector,
        target_label='person',
        acquisition_frames=acquisition_frames,
        lost_timeout_seconds=lost_timeout_seconds,
        scan_pan_speed=0.15
    )

def test_behavior_starts_idle() -> None:
    behavior = make_behavior()
    result = behavior.update([])
    assert result.state == BehaviorState.IDLE
    assert result.target is None
    assert result.lock_indicator_on is False
    assert result.requested_scan_pan_speed == 0.0

def test_start_scanning_requests_pan_speed() -> None:
    behavior = make_behavior()
    behavior.start_scanning()

    result = behavior.update([])

    assert result.state == BehaviorState.SCANNING
    assert result.target is None
    assert result.lock_indicator_on is False
    assert result.requested_scan_pan_speed == 0.15

def test_detection_enters_acquiring() -> None:
    behavior = make_behavior()
    behavior.start_scanning()
    person = make_person()
    result = behavior.update([person])

    assert result.state == BehaviorState.ACQUIRING
    assert result.lock_indicator_on is False
    assert result.requested_scan_pan_speed == 0.0

def test_target_locks_after_required_frames() -> None:
    behavior = make_behavior(acquisition_frames=3)
    behavior.start_scanning()
    person = make_person(track_id=7)
    first = behavior.update([person])
    second = behavior.update([person])
    third = behavior.update([person])

    assert first.state == BehaviorState.ACQUIRING
    assert second.state == BehaviorState.ACQUIRING
    assert third.state == BehaviorState.LOCKED

    assert third.target is not None
    assert third.target.track_id == 7
    assert third.lock_indicator_on is True
    assert third.requested_scan_pan_speed == 0.0

def test_acquisition_resets_if_target_disappears() -> None:
    behavior = make_behavior(acquisition_frames=3)
    behavior.start_scanning()
    person = make_person()
    behavior.update([person])
    result = behavior.update([])

    assert result.state == BehaviorState.SCANNING
    assert result.target is None
    assert result.lock_indicator_on is False

def test_largest_person_is_acquired() -> None:
    behavior = make_behavior(acquisition_frames=1)
    behavior.start_scanning()
    small_person = make_person(track_id=1, size=50)
    large_person = make_person(track_id=2, size=150)

    #first update enters acquiring
    behavior.update([small_person,large_person])
    #next update confirms acquisition
    result = behavior.update([small_person,large_person])
    
    assert result.state == BehaviorState.LOCKED
    assert result.target is not None
    assert result.target.track_id == 2

def test_locked_target_entering_lost_state() -> None:
    behavior = make_behavior(acquisition_frames=1)
    behavior.start_scanning()
    person = make_person(track_id=4)
    behavior.update([person])
    behavior.update([person])
    result = behavior.update([])

    assert result.state == BehaviorState.LOST
    assert result.target is None
    assert result.lock_indicator_on is False
    assert result.requested_scan_pan_speed == 0.0

def test_lost_target_can_be_recovered() -> None:
    behavior = make_behavior(acquisition_frames=1)
    behavior.start_scanning()
    person = make_person(track_id=4)
    behavior.update([person])
    behavior.update([person])
    behavior.update([])
    result = behavior.update([person])

    assert result.state == BehaviorState.LOCKED
    assert result.target is not None
    assert result.target.track_id == 4
    assert result.lock_indicator_on is True

def test_lost_timeout_resumes_scanning() -> None:
    behavior = make_behavior(acquisition_frames=1,lost_timeout_seconds=0.0)
    behavior.start_scanning()
    person = make_person(track_id=4)
    behavior.update([person])
    behavior.update([person])
    lost = behavior.update([])
    scanning = behavior.update([])

    assert lost.state == BehaviorState.LOST
    assert scanning.state == BehaviorState.SCANNING
    assert scanning.requested_scan_pan_speed == 0.15

def test_manual_lock() -> None:
    behavior = make_behavior()
    person = make_person(track_id=9)
    behavior.lock_manually(person)
    result = behavior.update([person])

    assert result.state == BehaviorState.LOCKED
    assert result.target is not None
    assert result.target.track_id==9
    assert result.lock_indicator_on is True

def test_stop_returns_to_idle() -> None:
    behavior = make_behavior()
    behavior.start_scanning()
    behavior.stop()
    result = behavior.update([])

    assert result.state == BehaviorState.IDLE
    assert result.target is None
    assert result.lock_indicator_on is False
    assert result.requested_scan_pan_speed == 0.0


def test_manual_selected_target_timeout_never_switches_to_another_person():
    behavior = make_behavior(lost_timeout_seconds=0)
    behavior.lock_manually(make_person(9))
    behavior.update([])
    result = behavior.update([make_person(10)])
    assert result.state == BehaviorState.IDLE
    assert result.target is None
    assert result.requested_scan_pan_speed == 0
