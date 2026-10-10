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
        prediction_frames=1.0,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    initial = tracker.start(image, (10, 20, 30, 40))
    updated = tracker.update(image)
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
