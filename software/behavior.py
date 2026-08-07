#systems behavior - decides whether camera should sit still, scan area, investigate target, follow confirmed target, wait for temp lost target, resume scanning

from __future__ import annotations 
from dataclasses import dataclass
from enum import Enum, auto
from time import monotonic
from software.detector import Detection
from software.tracker import TargetSelector

class BehaviorState(Enum):
    IDLE = auto()
    SCANNING = auto()
    ACQUIRING = auto()
    LOCKED = auto()
    LOST = auto()

@dataclass(frozen=True)
class BehaviorResult:
    state: BehaviorState
    target: Detection | None
    lock_indicator_on: bool
    requested_scan_pan_speed: float

class TrackingBehavior: 
    def __init__(self, selector: TargetSelector, target_label: str = "person", acquisition_frames: int=3, lost_timeout_seconds: float = 1.0, scan_pan_speed: float = 0.15) -> None:
        self.selector = selector
        self.target_label = target_label.lower()
        self.acquisition_frames = acquisition_frames
        self.lost_timeout_seconds = lost_timeout_seconds
        self.scan_pan_speed = scan_pan_speed
        
        self.state = BehaviorState.IDLE

        self._candidate_track_id: int | None = None
        self._candidate_frames = 0

        self._last_seen_time: float | None = None
        

        
        

    def start_scanning(self) -> None:
        self.selector.clear()

        self._candidate_track_id = None
        self._candidate_frames = 0
        self._last_seen_time = None
    
        self.state = BehaviorState.SCANNING

    def stop(self) -> None:
        self.selector.clear()
        self._candidate_track_id = None
        self._candidate_frames=0
        self._last_seen_time = None
        self.state = BehaviorState.IDLE

    def lock_manually(self, target: Detection) -> None:
        if target.track_id is None:
            return

        self.selector.selected_track_id= target.track_id
        self._last_seen_time = monotonic()
        self._candidate_track_id = None
        self._candidate_frames = 0
        self.state = BehaviorState.LOCKED

    def update(self, tracked_objects: list[Detection]) -> BehaviorResult:
        now = monotonic() 

        if self.state == BehaviorState.IDLE:
            return self._make_result(target=None, indicator=False)

        if self.state == BehaviorState.SCANNING:
            candidate = self._select_candidate(tracked_objects)

            if candidate is not None:
                self._candidate_track_id = candidate.track_id
                self._candidate_frames = 1
                self.state = BehaviorState.ACQUIRING

            return self._make_result(target=None, indicator=False)

        if self.state == BehaviorState.ACQUIRING:
            candidate = self._find_track(tracked_objects,self._candidate_track_id)
            if candidate is None:
                self._candidate_track_id = None
                self._candidate_frames = 0
                self.state = BehaviorState.SCANNING
                return self._make_result(target=None, indicator=False)

            self._candidate_frames +=1

            if self._candidate_frames >= self.acquisition_frames:
                self.selector.selected_track_id = (candidate.track_id)
                self._last_seen_time = now
                self._candidate_track_id = None
                self._candidate_frames = 0

                self.state = BehaviorState.LOCKED
                return self._make_result(target=candidate,indicator=True)
            return self._make_result(target=None,indicator=False)

        if self.state == BehaviorState.LOCKED:
            target = self.selector.get_selected(tracked_objects)
            if target is not None:
                self._last_seen_time = now
                return self._make_result(target=target, indicator=True)
            self.state = BehaviorState.LOST
            return self._make_result(target=None, indicator=False)

        if self.state == BehaviorState.LOST:
            target = self.selector.get_selected(tracked_objects)
            if target is not None:
                self._last_seen_time = now
                self.state = BehaviorState.LOCKED
                return self._make_result(target=target, indicator=True)

            if (self._last_seen_time is None or now - self._last_seen_time >= self.lost_timeout_seconds):
                self.start_scanning()
            return self._make_result(target=None, indicator=False)

        return self._make_result(target=None, indicator=False)

    def _select_candidate(self, tracked_objects: list[Detection]) -> Detection | None:
        candidates = [
            tracked_object
            for tracked_object in tracked_objects
            if (
                tracked_object.label.lower() == self.target_label and tracked_object.track_id is not None
            )
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda tracked_object: tracked_object.area)

    @staticmethod
    def _find_track(tracked_objects: list[Detection], track_id: int | None) -> Detection | None:
        if track_id is None:
            return None
        
        for tracked_object in tracked_objects:
            if tracked_object.track_id == track_id:
                return tracked_object
        return None

    def _make_result(self, target:Detection | None, indicator: bool) -> BehaviorResult:
        if self.state == BehaviorState.SCANNING:
            requested_scan_pan_speed = self.scan_pan_speed
        else:
            requested_scan_pan_speed = 0.0

        return BehaviorResult(state=self.state,target=target,lock_indicator_on=indicator,requested_scan_pan_speed=requested_scan_pan_speed)
        

