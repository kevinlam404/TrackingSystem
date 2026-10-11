from __future__ import annotations

from collections import deque
from threading import Condition, Event, Lock, Thread
from time import monotonic_ns
from uuid import uuid4
from base64 import b64encode
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template


class Dashboard:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self._condition = Condition()
        self._lock = Lock()
        self._jpeg: bytes | None = None
        self._status: dict[str, Any] = {}
        # Bound retained ROI frames if inference is blocked or a client floods commands.
        self._commands: deque[dict[str, Any]] = deque(maxlen=32)
        self._frame_size: tuple[int, int] | None = None
        self.motion_action = None
        self.motion_status = None
        self._preview_stop = Event()
        self._preview_thread = None
        self._annotations = []
        self._annotation_time_ns = 0
        self._raw_frame = None
        self._snapshots = {}
        self._selection_epoch = 0
        self._published_at_ns = 0
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 4096
        self._routes()

    def _routes(self) -> None:
        self.app.add_url_rule("/", "index", lambda: render_template("dashboard.html"))
        self.app.add_url_rule("/video_feed", "video", self._video)
        self.app.add_url_rule("/frame.jpg", "frame", self._frame)
        self.app.add_url_rule("/api/status", "status", self._get_status)
        self.app.add_url_rule("/api/action", "action", self._action, methods=["POST"])
        self.app.add_url_rule("/api/roi", "roi", self._roi, methods=["POST"])
        self.app.add_url_rule("/api/snapshot", "snapshot", self._snapshot, methods=["POST"])

    def start(self) -> None:
        Thread(
            target=lambda: self.app.run(
                host=self.host, port=self.port, threaded=True, use_reloader=False
            ),
            name="dashboard",
            daemon=True,
        ).start()

    def publish(self, image: np.ndarray, status: dict[str, Any], raw_frame=None) -> None:
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._published_at_ns = monotonic_ns()
            self._status.update(status)
            self._frame_size = (image.shape[1], image.shape[0])
            if raw_frame is not None:
                self._raw_frame = raw_frame
            self._condition.notify_all()

    def update_status(self, **status) -> None:
        with self._condition:
            self._status.update(status)

    def annotate(self, objects, timestamp_ns) -> None:
        with self._condition:
            self._annotations = list(objects)
            self._annotation_time_ns = timestamp_ns

    def start_preview(self, camera) -> None:
        """MJPEG keeps moving even while vision is slow, failing, or idle."""
        self._preview_stop.clear()
        def preview():
            sequence = -1
            last_frame_at = monotonic_ns()
            while not self._preview_stop.is_set():
                try:
                    frame = camera.wait_for_frame(sequence)
                    if frame is None:
                        if monotonic_ns() - last_frame_at > 1_000_000_000:
                            self.update_status(camera_state='error', camera_error='Camera is not producing fresh frames')
                        continue
                    sequence = frame.sequence
                    last_frame_at = monotonic_ns()
                    image = frame.image.copy()
                    with self._condition:
                        objects = self._annotations if 0 <= frame.timestamp_ns - self._annotation_time_ns < 250_000_000 else []
                    for obj in objects:
                        cv2.rectangle(image, (obj.x1, obj.y1), (obj.x2, obj.y2), (0, 255, 255), 2)
                        label = 'Selected object' if obj.track_id == -1 else f'{obj.label} ID {obj.track_id}'
                        cv2.putText(image, label, (obj.x1, max(18, obj.y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    self.publish(image, {'frame_sequence': sequence, 'camera_state': 'running',
                                         'camera_error': None, 'camera_frame_age_ms': round((monotonic_ns()-frame.timestamp_ns)/1e6, 1)}, raw_frame=frame)
                except Exception as error:
                    self.update_status(camera_state='error', camera_error=str(error))
                    sequence = -1
                    self._preview_stop.wait(0.2)
        self._preview_thread = Thread(target=preview, name='camera-preview', daemon=True)
        self._preview_thread.start()

    def stop_preview(self) -> None:
        self._preview_stop.set()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=2.0)

    def pop_commands(self) -> list[dict[str, Any]]:
        with self._lock:
            commands = list(self._commands)
            self._commands.clear()
            return commands

    def _queue(self, command: dict[str, Any]) -> None:
        with self._lock:
            self._commands.append(command)

    def _get_status(self):
        with self._condition:
            status = dict(self._status)
        if self.motion_status is not None:
            status.update(self.motion_status())
        return jsonify(status)

    def _frame(self):
        """Finite JPEG fallback for browsers/tunnels that fail to display MJPEG."""
        with self._condition:
            jpeg = self._jpeg
            fresh = monotonic_ns()-self._published_at_ns < 2_000_000_000
        if jpeg is None or not fresh:
            return jsonify(error='No fresh video frame. Check camera status and source.'), 503
        return Response(jpeg, mimetype='image/jpeg', headers={'Cache-Control': 'no-store'})

    def _action(self):
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error='Expected a JSON object'), 400
        action = str(data.get("action", ""))
        if action not in {"scan", "auto", "clear", "center", "stop", "estop", "reset_estop", "previous", "next"}:
            return jsonify(error="invalid action"), 400
        with self._condition:
            self._selection_epoch += 1
            self._snapshots.clear()
        if self.motion_action is not None:
            self.motion_action(action)
        self._queue({"type": "action", "action": action})
        return jsonify(ok=True)

    def selection_is_current(self, command):
        with self._condition:
            return command.get('selection_epoch', self._selection_epoch) == self._selection_epoch

    def _snapshot(self):
        if self.motion_action is not None:
            self.motion_action('stop')
        self._queue({'type': 'action', 'action': 'stop'})
        with self._condition:
            frame = self._raw_frame
            if frame is None or self._status.get('camera_state') == 'error' or monotonic_ns()-frame.timestamp_ns > 1_000_000_000:
                return jsonify(error='No fresh camera frame. Wait for the camera to reconnect.'), 409
            # Only a few short-lived raw snapshots: bounded memory on the Pi.
            now = monotonic_ns()
            self._snapshots = {key: value for key, value in self._snapshots.items() if now-value[0] < 30_000_000_000}
            while len(self._snapshots) >= 4:
                self._snapshots.pop(next(iter(self._snapshots)))
            token = uuid4().hex
            self._snapshots[token] = (now, frame, self._selection_epoch)
        success, encoded = cv2.imencode('.jpg', frame.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            return jsonify(error='Unable to encode selection snapshot'), 503
        return jsonify(snapshot_id=token, image='data:image/jpeg;base64,' + b64encode(encoded).decode('ascii'))

    def _roi(self):
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify(error='Expected a JSON object'), 400
        try:
            roi = tuple(int(data[key]) for key in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError, OverflowError):
            return jsonify(error="invalid ROI"), 400
        if roi[2] < 8 or roi[3] < 8:
            return jsonify(error="ROI is too small"), 400
        with self._condition:
            size = self._frame_size
            camera_failed = (self._status.get('camera_state') in ('starting', 'error')
                             or monotonic_ns() - self._published_at_ns > 1_000_000_000)
        if size is None or camera_failed:
            return jsonify(error="Wait for a camera frame before selecting an object"), 409
        if roi[0] < 0 or roi[1] < 0 or roi[0] + roi[2] > size[0] or roi[1] + roi[3] > size[1]:
            return jsonify(error="Selection must stay inside the camera image"), 400
        command = {"type": "roi", "roi": roi}
        if 'snapshot_id' in data:
            token = data['snapshot_id']
            if not isinstance(token, str):
                return jsonify(error='Invalid snapshot ID'), 400
            with self._condition:
                saved = self._snapshots.get(token)
                if saved is None or monotonic_ns()-saved[0] > 30_000_000_000:
                    return jsonify(error='Selection expired. Take a new snapshot.'), 409
                selected_frame = saved[1]
                height, width = selected_frame.image.shape[:2]
                if roi[0]+roi[2] > width or roi[1]+roi[3] > height:
                    return jsonify(error='Selection must stay inside the snapshot'), 400
                command['frame'] = selected_frame
                command['selection_epoch'] = saved[2]
                self._snapshots.pop(token, None)
        if self.motion_action is not None:
            self.motion_action('stop')
        self._queue(command)
        return jsonify(ok=True)

    def _video(self) -> Response:
        def frames():
            last: bytes | None = None
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._jpeg is not None and self._jpeg is not last, timeout=1.0)
                    current = self._jpeg
                if current is None or current is last:
                    continue
                last = current
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + current + b"\r\n"
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
