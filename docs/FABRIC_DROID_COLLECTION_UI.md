# Fabric-DROID collection UI

The UI lives entirely in the DROID repository. It reads no code from
Fabric-Omni and never writes to Fabric-Omni or Clothumi.

Launch it with:

```bash
cd /home/suhang/projects/droid
/home/suhang/anaconda3/envs/clothumi/bin/python tools/fabric_droid_collection_ui.py
```

The current verified camera mapping is:

- wrist D435: `338622072599`
- third-person D435: `102422075665`
- GelSight Mini: `2DWF0RJM`

The two D435 assignments can be swapped in the UI after inspecting the live
views. RealSense cameras are opened by the SDK serial. GelSight is opened by
its `/dev/v4l/by-id` link, so `/dev/videoN` renumbering does not change the
saved assignment. The UI refuses to substitute the laptop webcam when the
known GelSight device is absent.

ATI is mandatory in this UI. **开始录制** remains disabled until all three
video streams have fresh frames and ATI has a fresh, finite, nonzero wrench
window. A camera stall, ATI timeout, all-zero ATI window, device-status error,
or video-writer failure during recording automatically stops teleoperation and
leaves the episode marked incomplete. Episode-local maximum camera/ATI gaps are
latched, so a stream cannot recover later and hide an earlier interruption.

The active task no longer uses tray color or left/right labels. Every new
episode stores:

- `destination_tray = "target_tray"`
- `task_instruction = "Inspect the fabric and place it in the target tray."`

The UI records the real DROID robot trajectory together with RGB, GelSight,
timestamps, events, device assignment, calibration, all ATI rows, and the
capture report. It never invents robot state/action values.

## Franka setup controls

The UI launches robot commands with the dedicated
`droid-polymetis-client` Python environment:

- **开启自由拖动** performs a read-only preflight and then starts low
  Cartesian impedance immediately, without an input dialog.
- **记录 Home** atomically overwrites
  `configs/robot/franka_home_pose.json` with the current pose, then terminates
  the UI-owned low-impedance policy.
- **回到 Home** performs another preflight and then immediately uses the
  guarded ten-second minimum-jerk joint-space trajectory.

If low impedance is active when **开始录制** is pressed, the UI first sends
SIGINT to the freedrive subprocess, waits until that UI-owned policy has
actually terminated, and only then starts sensor recording. No robot action
confirmation dialog is shown.

After sensor warmup succeeds, the same button starts Quest teleoperation. The
right-controller side grip remains the deadman: hold it to move, release it to
hold. Recording itself continues for the whole episode; releasing the side grip
does not discard sensor or robot-state samples.

The UI never terminates a policy it did not start.

## Dataset browser

Use **选择** beside the output path to choose a dataset root. The episode table
refreshes automatically and shows complete/incomplete status, duration, camera
frame counts, ATI sample count/rate, swatch ID, and split. Double-click a row
or use **打开目录** to inspect the saved episode.
