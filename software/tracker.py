from __future__ import annotations
from software.detector import Detection

class TargetSelector:
    def __init__(self) -> None:
        self.selected_track_id: int | None = None

    def get_selected(self, tracked_objects: list[Detection]) -> Detection | None:
        if self.selected_track_id is None:
            return None

        for tracked_object in tracked_objects:
            if tracked_object.track_id == self.selected_track_id:
                return tracked_object

        return None

    def select_largest_person(self, tracked_objects: list[Detection]) -> Detection | None:
        people = [
            tracked_object
            for tracked_object in tracked_objects if tracked_object.label.lower() == "person"
        ]

        if not people:
            return None

        selected = max(people, key=lambda tracked_object: tracked_object.area)
        self.selected_track_id = selected.track_id
        return selected

    def select_at_point(self, x:int,y:int,tracked_objects: list[Detection]) -> Detection | None:
        candidates = [
            tracked_object
            for tracked_object in tracked_objects if tracked_object.contains_point(x,y)
        ]
        if not candidates:
            return None

        #If boxes overlap, select the smallest box under the cursor
        selected = min(candidates, key=lambda tracked_object: tracked_object.area)
        self.selected_track_id = selected.track_id
        return selected

    def cycle(self, tracked_objects: list[Detection], direction: int = 1) -> Detection | None:
        available=[
            tracked_object for tracked_object in tracked_objects if tracked_object.track_id is not None
        ]

        if not available:
            return None

        available.sort(key=lambda tracked_object: tracked_object.track_id)
        
        track_ids = [
            tracked_object.track_id
            for tracked_object in available
        ]

        if self.selected_track_id not in track_ids:
            selected=available[0]
        else:
            current_index = track_ids.index(self.selected_track_id)
            next_index = (current_index + direction) % len(available)
            selected = available[next_index]

        self.selected_track_id = selected.track_id
        return selected
    
    def clear(self) -> None:
        self.selected_track_id = None