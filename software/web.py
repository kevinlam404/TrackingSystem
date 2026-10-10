from __future__ import annotations

from collections import deque
from threading import Condition, Lock, Thread
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request


PAGE = """<!doctype html>
<html><head><meta name="viewport" content="width=device-width">
<title>Tracking Gimbal</title>
<style>
body{font-family:system-ui;background:#111;color:#eee;margin:20px}button{margin:4px;padding:10px}
#stage{position:relative;display:inline-block;max-width:100%}#video{max-width:100%;display:block}
#box{position:absolute;border:2px solid #ff0;display:none;pointer-events:none}pre{background:#222;padding:12px}
</style></head><body>
<h2>Tracking Gimbal</h2><div id="stage"><img id="video" src="/video_feed"><div id="box"></div></div>
<div><button onclick="act('scan')">Scan</button><button onclick="act('auto')">Auto person</button>
<button onclick="act('clear')">Clear</button><button onclick="act('center')">Center</button>
<button onclick="act('stop')">Stop</button><button onclick="act('estop')">E-STOP</button></div>
<p>Drag a box on the video to track any visible object.</p><pre id="status">connecting...</pre>
<script>
const img=document.getElementById('video'), box=document.getElementById('box'); let start=null;
function pos(e){const r=img.getBoundingClientRect();return{x:e.clientX-r.left,y:e.clientY-r.top,r}}
img.onpointerdown=e=>{start=pos(e);box.style.display='block';box.style.left=start.x+'px';box.style.top=start.y+'px'};
img.onpointermove=e=>{if(!start)return;const p=pos(e),x=Math.min(start.x,p.x),y=Math.min(start.y,p.y);
box.style.left=x+'px';box.style.top=y+'px';box.style.width=Math.abs(p.x-start.x)+'px';box.style.height=Math.abs(p.y-start.y)+'px'};
img.onpointerup=async e=>{if(!start)return;const p=pos(e),sx=img.naturalWidth/p.r.width,sy=img.naturalHeight/p.r.height;
const body={x:Math.round(Math.min(start.x,p.x)*sx),y:Math.round(Math.min(start.y,p.y)*sy),
width:Math.round(Math.abs(p.x-start.x)*sx),height:Math.round(Math.abs(p.y-start.y)*sy)};
start=null;box.style.display='none';await fetch('/api/roi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})};
async function act(action){await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})})}
setInterval(async()=>{try{document.getElementById('status').textContent=JSON.stringify(await (await fetch('/api/status')).json(),null,2)}catch(e){}},500);
</script></body></html>"""


class Dashboard:
    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self._condition = Condition()
        self._lock = Lock()
        self._jpeg: bytes | None = None
        self._status: dict[str, Any] = {}
        self._commands: deque[dict[str, Any]] = deque()
        self.app = Flask(__name__)
        self._routes()

    def _routes(self) -> None:
        self.app.add_url_rule("/", "index", lambda: PAGE)
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
            self._status = dict(status)
            self._condition.notify_all()

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
            return jsonify(self._status)

    def _action(self):
        action = str((request.get_json(silent=True) or {}).get("action", ""))
        if action not in {"scan", "auto", "clear", "center", "stop", "estop"}:
            return jsonify(error="invalid action"), 400
        self._queue({"type": "action", "action": action})
        return jsonify(ok=True)

    def _roi(self):
        data = request.get_json(silent=True) or {}
        try:
            roi = tuple(int(data[key]) for key in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError):
            return jsonify(error="invalid ROI"), 400
        if roi[2] < 2 or roi[3] < 2:
            return jsonify(error="ROI is too small"), 400
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
