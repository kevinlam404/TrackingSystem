# TrackingSystem

Raspberry Pi 5 tracking camera with automatic person detection, manual CSRT
object tracking, a browser dashboard, and a two-axis servo gimbal.

## Start

Activate your existing virtual environment, then run from the repository:

```sh
python -m software.main --config config.yaml
```

The app starts **idle**: YOLO runs and displays all available detected track
boxes, while the motors remain stopped. Choose Previous/Next to follow one,
or Auto person / Scan for automatic acquisition. Once locked, only the selected
box is shown. Stop returns to idle browsing; manual drawn-box tracking uses
CSRT instead of YOLO.

Check these settings before starting:

- Ribbon-cable Pi camera: `camera.source: picamera`.
- USB webcam: `camera.source: webcam` and the correct `camera.device`.
- Start with `gimbal.backend: simulated`. Physical mode requires calibrated
  pulses, verified directions/limits, and an accessible servo-power switch.
- `interface.local_preview: false` for SSH/headless operation.

The browser listens on the Pi's localhost by default. From your own computer:

```sh
ssh -L 8080:127.0.0.1:8080 kevinlam@192.168.4.49
```

Keep the SSH session open, then visit `http://127.0.0.1:8080` locally.
The tracking app must also be running on the Pi.

## Blank video

Camera capture and MJPEG preview are independent of model inference. Camera
errors are shown in the dashboard and terminal; the app retries camera startup
every two seconds. Inference failures stop motion but do not stop the video.
The status panel distinguishes camera connection from tracking state.
The browser automatically falls back to individual JPEG frames if the MJPEG
connection cannot display video. Reconnect video restarts the connection.
Status-request errors do not cover an already decoded image. `/frame.jpg`
returns one fresh JPEG (or a clear 503 error if no recent frame exists), which
can help distinguish a camera failure from a browser streaming failure.

Check `camera.source` first. A working dashboard does not prove that the
selected camera is available. Use `python -m tools.test_video --source picamera`
or `--source webcam` only when a graphical desktop is available: that utility
opens an OpenCV window and is not the headless browser interface.

## Controls

- Auto person / Scan starts automatic person acquisition.
- Previous/Next (or P/N in the browser) cycles current detected track IDs.
- A manually chosen detected ID returns to idle after the loss timeout; it
  never silently switches to scanning for another person. Auto mode may rescan.
- Select object stops motion and freezes a clean snapshot. Draw and confirm a
  box, or Refresh snapshot / Cancel. Selection expires after 30 seconds. The
  tracker finds the selected appearance in current video before moving motors;
  it does not assume the object stayed at the old snapshot coordinates.
- Stop halts tracking. Emergency stop is latched until reset; reset alone
  never starts motion.
- Center tilt stops tracking and centers tilt; unlimited pan has no angle feedback.

## Verification

```sh
python -m pytest -q
python -m tools.hardware_smoke_test
```

The smoke test is simulated unless explicitly given `--physical --i-understand`.
`tools/servo_calibration.py` is a hardware-only calibration utility; do not run
it simultaneously with the tracking app. See [control and tracking notes](docs/control-and-roi.md).

Automated tests exercise simulated hardware and synthetic video, not the actual
Pi camera, I2C wiring, servo load, or thermal performance. Those require bench
tests. A software stop cannot neutralize a PCA9685 after the process dies or
I2C stops responding: retain a physical servo-power cutoff.

### Servo safety behavior

Normal scan/tracking pan commands ramp at `control.pan_acceleration_per_second`
(normalized speed per second, not measured degrees/s²). Stale frames, Stop,
and emergency stop bypass the ramp and request neutral immediately.
PCA9685 `disable()` rejects movement but keeps calibrated pan-neutral PWM
and the last tilt command active to avoid dropping the camera. It does NOT
remove power; use the physical servo-power switch before touching the assembly.
An I2C fault may prevent neutral from reaching the board, and a process crash
may leave the last PWM running. Browser stop requests remain responsive during
blocked I/O, but cannot interrupt the bus operation or guarantee physical stop.
Calibration rejects pulses outside 900–2100 us for every input path, including
`+`, `-`, and startup. These are tool guardrails, NOT proof of safe mechanical
travel: start near neutral and determine narrower limits for your assembly.

### Shared runtime and validation

`python -m tools.test_tracker` uses the same tracking and motor runtime as
`software.main`, with simulated motors and a local OpenCV window. Click a box
to select it, P/N to cycle, A/S to scan, X to stop, and Q/Escape to exit.
This removes the separate legacy control loop; it never enables physical hardware.
`tools.test_detector` is detection-only and does not require confirmed track IDs.
Configuration and nonfinite motor inputs are rejected instead of silently
producing extreme commands. NCNN and ByteTrack's LAP dependency are included
in `requirements.txt` to avoid first-run dependency downloads.

Displayed frame age begins when the capture API returns the image; it does not
include sensor exposure or hidden camera buffering. It is not a complete
capture-to-motion latency measurement. Pi hardware testing is still required.
The dashboard is intended for localhost/SSH tunneling, not public exposure:
it has no authentication. Camera/servo power faults need physical safeguards.
