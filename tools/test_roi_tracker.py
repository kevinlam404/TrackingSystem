import numpy as np

from software.roi_tracker import ROITracker


class FakeTracker:
    def __init__(self):
        self.box = (10.0, 20.0, 30.0, 40.0)

    def init(self, image, box):
        self.box = tuple(float(value) for value in box)
        return True

    def update(self, image):
        x, y, width, height = self.box
        self.box = (x + 2.0, y + 1.0, width, height)
        return True, self.box


def test_roi_tracker_starts_and_updates():
    tracker = ROITracker(
        tracker_factory=FakeTracker,
        velocity_smoothing=0.0,
        prediction_seconds=0.1,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    initial = tracker.start(image, (10, 20, 30, 40), timestamp=10)
    updated = tracker.update(image, timestamp=10.1)
    assert initial.center == (25, 40)
    assert updated is not None
    assert updated.center == (29, 42)


def test_roi_tracker_rejects_empty_box():
    tracker = ROITracker(tracker_factory=FakeTracker)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        tracker.start(image, (0, 0, 0, 10))
    except ValueError:
        return
    raise AssertionError('Expected invalid ROI to be rejected')


class MissingTracker(FakeTracker):
    def update(self, image):
        return False, self.box


def textured_frame(x=30, y=30, duplicate=False):
    image = np.zeros((140, 180, 3), dtype=np.uint8)
    patch = np.random.default_rng(42).integers(0, 255, (20, 20, 3), dtype=np.uint8)
    image[y:y+20, x:x+20] = patch
    if duplicate:
        image[30:50, 65:85] = patch
    return image


def test_short_loss_reacquires_appearance_without_guessing_motor_target():
    tracker = ROITracker(tracker_factory=MissingTracker)
    tracker.start(textured_frame(), (30, 30, 20, 20), timestamp=10)
    assert tracker.update(textured_frame(35), timestamp=10.1) is None
    assert tracker.active and tracker.state == 'REACQUIRING'
    assert tracker.update(textured_frame(35), timestamp=10.2) is None
    found = tracker.update(textured_frame(35), timestamp=10.3)
    assert found is not None
    assert found.center == (45, 40)
    assert tracker.state == 'LOCKED'


def test_ambiguous_match_not_reacquired():
    tracker = ROITracker(tracker_factory=MissingTracker)
    tracker.start(textured_frame(), (30, 30, 20, 20), timestamp=10)
    tracker.update(textured_frame(), timestamp=10.1)
    assert tracker.update(textured_frame(30, duplicate=True), timestamp=10.2) is None


def test_loss_times_out_without_switching_to_person_mode():
    tracker = ROITracker(tracker_factory=MissingTracker, reacquire_timeout_seconds=0.5)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker.start(image, (10, 20, 30, 40), timestamp=10)
    assert tracker.update(image, timestamp=10.1) is None
    assert tracker.update(image, timestamp=10.7) is None
    assert not tracker.active and tracker.state == 'LOST'


def test_prediction_uses_elapsed_time():
    tracker = ROITracker(tracker_factory=FakeTracker, velocity_smoothing=0,
                         prediction_seconds=0.05)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker.start(image, (10, 20, 30, 40), timestamp=10)
    found = tracker.update(image, timestamp=10.1)
    assert found.center == (28, 42)


def test_appearance_change_does_not_veto_successful_csrt():
    tracker = ROITracker(tracker_factory=FakeTracker)
    tracker.start(textured_frame(), (30, 30, 20, 20), timestamp=10)
    # Entirely changed patch previously caused an immediate loss despite success.
    changed = np.random.default_rng(7).integers(0, 255, (140, 180, 3), dtype=np.uint8)
    for i in range(5):
        assert tracker.update(changed, timestamp=10.1 + i * 0.1) is not None
        assert tracker.state == 'LOCKED'


def test_partial_frame_edge_does_not_immediately_lose_track():
    class EdgeTracker(FakeTracker):
        def update(self, image):
            return True, (-3, 20, 30, 40)
    tracker = ROITracker(tracker_factory=EdgeTracker)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker.start(image, (0, 20, 30, 40), timestamp=10)
    target = tracker.update(image, timestamp=10.1)
    assert target is not None and target.x1 >= 0
    assert tracker.state == 'LOCKED'


def test_unrelated_scene_does_not_reacquire():
    tracker = ROITracker(tracker_factory=MissingTracker)
    tracker.start(textured_frame(), (30, 30, 20, 20), timestamp=10)
    tracker.update(textured_frame(), timestamp=10.1)
    unrelated = np.random.default_rng(100).integers(0, 255, (140, 180, 3), dtype=np.uint8)
    for i in range(3):
        assert tracker.update(unrelated, timestamp=10.2+i*0.1) is None


def test_rotated_feature_geometry_can_recover_without_template_match():
    import cv2
    patch = np.random.default_rng(10).integers(0, 255, (100, 100), dtype=np.uint8)
    original = np.zeros((220, 260, 3), dtype=np.uint8)
    original[50:150, 60:160] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    tracker = ROITracker(tracker_factory=MissingTracker, appearance_threshold=0.99)
    tracker.start(original, (60, 50, 100, 100), timestamp=10)
    rotated = cv2.warpAffine(patch, cv2.getRotationMatrix2D((50, 50), 12, 1.0), (100, 100))
    moved = np.zeros_like(original)
    moved[60:160, 70:170] = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    tracker.update(moved, timestamp=10.1)
    tracker.update(moved, timestamp=10.2)
    found = tracker.update(moved, timestamp=10.3)
    assert found is not None
    assert abs(found.center[0]-120) < 8 and abs(found.center[1]-110) < 8
