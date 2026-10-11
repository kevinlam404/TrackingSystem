# Independent motor control and manual tracking

Run `python -m software.main` as before. No new packages are required.

The motor worker owns all servo writes and targets `control.update_rate_hz`
(40 Hz by default). Vision publishes only the latest motion intent, tagged
with the original camera timestamp. Once that observation is older than
`control.stale_frame_timeout_seconds`, the worker stops pan and holds tilt,
even if inference is still processing. This is a Python thread, not a hard
real-time guarantee; measure its cadence on the Pi under load.

Browser Stop and Emergency stop reach the motor worker directly without
waiting for inference. Ordinary Stop rejects late results until a new mode
is selected. Emergency stop is latched: press Reset emergency stop, then
choose a mode to resume. Hardware I/O faults remain latched until app restart.
Center tilt is a one-shot command, not pan-position homing.

Manual tracking still uses CSRT, with time-based smoothed velocity prediction
(`roi_tracking.prediction_seconds`, capped at 150 ms and half a box). It checks
appearance against the selected patch to reject obvious drift. After loss,
motion holds while a bounded local template search tries three nearby scales
for up to `reacquire_timeout_seconds`. Matches must exceed
`appearance_threshold` and be distinguishable from competing matches.
Low-texture patches cannot be reliably reacquired. Expiration requires a new
manual selection; it never silently switches to a person.

This is not semantic object recognition or guaranteed identity recovery.
Rotation, lighting changes, similar objects, and long occlusion can defeat
template matching. No motor commands are produced from an unconfirmed
predicted target during occlusion.

## Bench validation (start with simulated backend)

- Run `python -m pytest -q`.
- Select an object; move it slowly and briefly obscure it, then reveal it nearby.
- Try two identical objects: ambiguous recovery should remain held.
- Interrupt camera input and stall inference: pan should stop after the stale timeout.
- Press Emergency stop during tracking; Scan must not restart motion until reset.
- Inspect `motor_ticks`, `motor_stale`, and `fault` in browser diagnostics.
- Verify real control cadence, CPU temperature, calibration, and physical stop
  behavior before operating real servos unattended.

An I2C call can itself hang or fail, and a dead process cannot send a neutral
pulse to a PCA9685 that keeps its previous PWM. The physical servo-power
switch remains essential; this software is not a hardware safety interlock.
