# TrackingSystem

Raspberry Pi 5 tracking camera with automatic person detection, manual CSRT
object tracking, a browser dashboard, and a two-axis servo gimbal.

## Start

Activate your existing virtual environment, then run from the repository:

```sh
python -m software.main --config config.yaml
```

The app starts **idle**: video is live, but no model loads and no motion starts
until you choose Auto person, Scan, Previous/Next, or a manual selection.

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

Check `camera.source` first. A working dashboard does not prove that the
selected camera is available. Use `python -m tools.test_video --source picamera`
or `--source webcam` only when a graphical desktop is available: that utility
opens an OpenCV window and is not the headless browser interface.

## Controls

- Auto person / Scan starts automatic person acquisition.
- Previous/Next (or P/N in the browser) cycles current detected track IDs.
- Select object stops motion, then lets you draw and confirm a box.
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
