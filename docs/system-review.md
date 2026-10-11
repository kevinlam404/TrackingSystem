# Whole-system review

## Changes

- Local tracker demo now uses the production runtime instead of a separate
  400-line camera/behavior/control loop. Mouse selection, target cycling,
  center marker, and target overlays remain available; motors are simulated.
- Detection-only demo has a working detector method. Detection and tracking
  share box conversion rather than duplicate conversion implementations.
- Removed unused largest-person selector; automatic acquisition already
  owns that decision in the behavior state machine.
- Manual YOLO selection no longer silently changes to another person after
  target-loss timeout. Automatic scanning retains its rescan behavior.
- Stop invalidates pending snapshot selections. A delayed confirmation cannot
  override a newer browser command. Snapshot consumption is atomic.
- Camera silence is reported and the runtime attempts restart. Partial Pi
  startup failures close the device; shutdown closes it even if stop fails.
- Removed redundant RGB-to-BGR swap for Picamera2 RGB888 capture.
- Configuration and nonfinite control inputs fail explicitly. Fixed exported
  model resolution mismatches fail with a helpful error before inference.
- Bounded browser command retention and request payload size.
- Selection coordinates use the displayed image, not the stage's minimum height.
- Added NCNN and LAP dependencies so first-run inference need not install them.
- Exported model's diagnostic script resolves assets relative to itself.

Servo bounds remain enforced in both control and driver layers intentionally:
these are independent safeguards, not redundant code to remove. Capture,
preview, and motor workers likewise serve different timing requirements.

## Verification

- Full pytest regression suite, Python compilation, static checks (Pyflakes),
  dependency consistency (`pip check`), and simulated hardware smoke test.
- Actual bundled NCNN detector on Ultralytics' installed bus image: detected
  people and a bus; ByteTrack kept consistent IDs across repeated frames.
- Headless Chrome against the actual Flask dashboard and synthetic camera:
  video display, target navigation, frozen-box selection with native CSRT,
  emergency-stop latch, and video reconnect. No JavaScript exceptions.

## Not verified / remaining limitations

- No physical Pi camera, I2C, MG996R/S51, load, stall, thermal, or power testing.
- Motor timing is best-effort Python threading, not hard real-time.
- PWM can continue after process death or bus failure; physical power cutoff
  is essential. Logical disable retains neutral pan and tilt torque.
- No encoders: speed/angle are commands, not measurements.
- Frame age excludes exposure and hidden capture buffering.
- CSRT and local appearance recovery cannot guarantee object identity under
  severe occlusion, rotation, or similar-looking objects. ByteTrack is not
  long-term identity recognition either.
- Dashboard has no authentication; use localhost and an SSH tunnel. Flask's
  development server is not intended for public deployment.
- Passing local tests is not a guarantee that every configuration and failure
  condition will work. Repeat verification on the Pi before physical operation.

No commits or pushes were made as part of this review.
