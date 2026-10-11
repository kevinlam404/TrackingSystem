from __future__ import annotations

from collections import deque
from threading import Condition, Event, Lock, Thread
from time import monotonic_ns
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
        self._commands: deque[dict[str, Any]] = deque()
        self._frame_size: tuple[int, int] | None = None
        self.motion_action = None
        self.motion_status = None
        self._preview_stop = Event()
        self._preview_thread = None
        self._annotations = []
        self._annotation_time_ns = 0
        self.app = Flask(__name__)
        self._routes()

    def _routes(self) -> None:
        self.app.add_url_rule("/", "index", lambda: render_template("dashboard.html"))
        self.app.add_url_rule("/video_feed", "video", self._video)
        self.app.add_url_rule("/api/status", "status", self._get_status)
        self.app.add_url_rule("/api/action", "action", self._action, methods=["POST"])
        self.app.add_url_rule("/api/roi", "roi", self._roi, methods=["POST"])

    def start(self) -> None:
        Thread(
            target=lambda: self.app.run(
                host=self.host, port=self.port, threaded=True, use_reloader=False
            ),
            name="dashboard",
            daemon=True,
        ).start()

    def publish(self, image: np.ndarray, status: dict[str, Any]) -> None:
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._status.update(status)
            self._frame_size = (image.shape[1], image.shape[0])
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
            while not self._preview_stop.is_set():
                try:
                    frame = camera.wait_for_frame(sequence)
                    if frame is None:
                        continue
                    sequence = frame.sequence
                    image = frame.image.copy()
                    with self._condition:
                        objects = self._annotations if 0 <= frame.timestamp_ns - self._annotation_time_ns < 250_000_000 else []
                    for obj in objects:
                        cv2.rectangle(image, (obj.x1, obj.y1), (obj.x2, obj.y2), (0, 255, 255), 2)
                        label = 'Selected object' if obj.track_id == -1 else f'{obj.label} ID {obj.track_id}'
                        cv2.putText(image, label, (obj.x1, max(18, obj.y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    self.publish(image, {'frame_sequence': sequence, 'camera_state': 'running',
                                         'camera_error': None, 'camera_frame_age_ms': round((monotonic_ns()-frame.timestamp_ns)/1e6, 1)})
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

    def _action(self):
        action = str((request.get_json(silent=True) or {}).get("action", ""))
        if action not in {"scan", "auto", "clear", "center", "stop", "estop", "reset_estop", "previous", "next"}:
            return jsonify(error="invalid action"), 400
        if self.motion_action is not None:
            self.motion_action(action)
        self._queue({"type": "action", "action": action})
        return jsonify(ok=True)

    def _roi(self):
        data = request.get_json(silent=True) or {}
        try:
            roi = tuple(int(data[key]) for key in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError):
            return jsonify(error="invalid ROI"), 400
        if roi[2] < 8 or roi[3] < 8:
            return jsonify(error="ROI is too small"), 400
        with self._condition:
            size = self._frame_size
            camera_failed = self._status.get('camera_state') in ('starting', 'error')
        if size is None or camera_failed:
            return jsonify(error="Wait for a camera frame before selecting an object"), 409
        if roi[0] < 0 or roi[1] < 0 or roi[0] + roi[2] > size[0] or roi[1] + roi[3] > size[1]:
            return jsonify(error="Selection must stay inside the camera image"), 400
        self._queue({"type": "roi", "roi": roi})
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
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
