# Fabric-DROID collection and π0.5 pipeline report

Date: 2026-07-25

## Scope and read-only boundary

All implementation changes are in `/home/suhang/projects/droid`. The following
repositories were read only:

| Repository | Path | Branch / commit at audit | State at audit |
|---|---|---|---|
| DROID | `/home/suhang/projects/droid` | `main` / `33ae6a67274f` | clean before this implementation |
| OpenPI | `/home/suhang/projects/openpi` | `main` / `15a9616a0094` | clean |
| Clothumi | `/home/suhang/projects/Clothumi` | `frabric-umi` / `4fdb40e93ff0` | 7 dirty/untracked entries, pre-existing |
| Octopi reference parent | `/home/suhang/projects/test_code/2` | `feature-testlabel` / `e3fb061d617f` | 58 dirty/untracked entries, pre-existing |
| Fabric-Omni | `/home/suhang/projects/fabric-omni` | untracked within dirty parent `/home/suhang/projects` | strictly read only |

Before and after Fabric-Omni bridge verification, a manifest of path, type,
size, mtime, and symlink target had the same SHA-256:

`7d97979eb4bec9a0fd31a16e1dae7402e4a9d4a545f3f076fe40484fcce62714`

The Final-170 checkpoint hash also remained:

`132aeab7a9f2e736ed5e8c83c86e3280ce413590c2b1738534c6d1a5dd0f7f0e`

No formatter, test runner, cache writer, or training command was run inside
Fabric-Omni. Python bytecode and Hugging Face caches were redirected to `/tmp`.

## Read-only audit findings

- DROID laptop entry: `scripts/main.py` constructs `RobotEnv`, `VRPolicy`,
  `DataCollecter`, and `RobotGUI`.
- Franka NUC entry: `scripts/server/run_server.py`; the ZeroRPC server owns
  `FrankaRobot`, which owns the Polymetis robot/gripper interfaces.
- The DROID laptop process is the sole action owner. The Fabric sidecar has no
  Franka connection and only receives lifecycle/action callbacks.
- DROID runs its control loop at 15 Hz by default. `RobotEnv.step` sends the
  action; the trajectory stores measured robot/gripper state separately from
  joint-velocity and commanded-gripper action.
- Success/failure comes from Oculus A/B controller state. A trajectory starts
  in the failure tree and the whole episode directory is renamed to success
  only after a successful close. Relabeling updates HDF5 attributes and moves
  the directory.
- Base DROID records ZED SVO streams through its camera wrapper. The trajectory
  writer excludes image/depth/pointcloud arrays from HDF5.
- `trajectory.h5` is created by `TrajectoryWriter`; post-processing later emits
  `metadata_<uuid>.json`.
- Clothumi's raw session code provided the host-arrival monotonic timestamp,
  producer/consumer thread, warmup, health, and rate/gap-reporting patterns.
- Octopi's `data_collect/local_recorder.py` provided the proven UVC + ATI
  parser/ZMQ pattern. Its no-decimation configuration (`capture_rate_hz: 0`,
  `conflate: false`) was retained conceptually. No source file was copied or
  edited in either reference repository.
- OpenPI's official converter uses 7-D joint velocity plus commanded gripper
  position. Current `DroidInputs` consumes external camera 1, left wrist,
  7-D joint state, 1-D measured gripper state, and prompt. For π0/π0.5 the
  right wrist image is masked.
- Current `pi05_droid_finetune` uses `action_dim=32`,
  `action_horizon=16`, the `pi05_droid` normalization assets, and the
  `pi05_droid` parameter checkpoint. The converter does not alter OpenPI.
- Fabric-Omni exposes `TeacherBV2(SingleObservationBatch)`. Final-170 produces
  M/S_obs/R/C/U, a 768-D global physical token, side posterior, availability,
  and auxiliary physical-response predictions. It is one-press,
  observed-surface-only, and does not use RGB in its physical path.

## Implemented architecture

```text
                         one robot-control owner
 Oculus -> DROID 15 Hz -> RobotEnv/Franka -> trajectory.h5
                  |
                  +-> optional lifecycle + before_action hooks
                         |-- ATI raw norm interlock before env.step
                         |-- F9 probe_complete / F10 release_time
                         `-- atomic sidecar close

 D435 thread -----------\
 wrist UVC thread -------+-> host time.monotonic_ns -> MP4 + timestamp NPZ
 GelSight UVC thread ----/
 ATI ZMQ thread (no conflate, no decimation) -> ati_raw.parquet

 episode -> validator -> segment manifests -> LeRobot -> OpenPI DroidInputs
         `-> 16-frame press -> frozen Final-170 -> physical-state NPZ cache
```

The sidecar starts only after every sensor has produced a fresh sample. Warmup
data is excluded from the episode time interval. A DROID hook factory stages
sidecar files separately, then atomically moves the tactile directory and new
camera files into the DROID-owned episode after `trajectory.h5` has closed.
Interrupted staging remains recoverable and is never presented as complete.

The original DROID path is unchanged when no `lifecycle_hook_factory` is
provided. Legacy wall-clock fields remain; new host-monotonic fields are added
alongside them.

## Episode schema

```text
episode_id/
├── trajectory.h5
├── metadata_<episode_id>.json
├── COMPLETE.json
├── recordings/
│   ├── MP4/
│   │   ├── exterior_image_1_left.mp4
│   │   └── wrist_image_left.mp4
│   └── timestamps/
│       ├── exterior_image_1_left.npz
│       └── wrist_image_left.npz
└── tactile/
    ├── gelsight_left.mp4
    ├── gelsight_left_timestamps.npy
    ├── gelsight_left_frame_metadata.npz
    ├── ati_raw.parquet
    ├── events.json
    ├── automatic_event_candidates.json       # optional offline output
    ├── calibration.json
    └── capture_report.json
```

ATI Parquet columns are `timestamp_monotonic_ns`, `timestamp_wall_ns`,
`sample_index`, `Fx`, `Fy`, `Fz`, `Tx`, `Ty`, `Tz`, `device_status`, and
`device_timestamp_ns`. Camera metadata contains frame index, host capture and
receive timestamps, optional device timestamp, serial/configuration, and
dropped-frame statistics.

`calibration.json` requires `T_ati_to_gripper`, `gripper_normal_axis`, and
`force_sign` before any normal-force-derived value is marked valid. Raw force
norm is allowed for an independent conservative interlock and event candidates,
but raw Fz is never treated as gripper-normal force.

Metadata includes the requested episode/task/source/swatch/split/start
pose/success/operator/session/robot/sensor/calibration/software fields. The
active task uses only the canonical `target_tray` label and contains no tray
color or direction class.

## Safety behavior

- Automated commands require `--dry-run`, `--record-only`, or
  `--robot-disabled`.
- `run_fabric_droid_teleop.py` does not import or construct `RobotEnv` unless
  both `--enable-robot` and `--preflight-confirmed` are supplied.
- The real sidecar must see healthy, advancing D435, wrist, GelSight, and ATI
  streams before recording begins.
- `ForceInterlockLifecycleHook` runs before `env.step`. Warning limits hold the
  gripper through an injected controller-specific callback; hard force/torque
  limits invoke abort and terminate the episode.
- `validate_robot_preflight` requires emergency-stop, controllable-mode, joint,
  workspace, and gripper-limit booleans from the DROID-owned robot interface.
- No normal-force closed loop exists in this implementation.
- F9 records `probe_complete`; F10 records `release_time`. Record-only CLI also
  accepts `p`/`r` followed by Enter with `--interactive-events`.

Before a real pilot, manually confirm Desk/estop state, Franka controllable
mode, joint/workspace/gripper bounds, ready pose, collision behavior, ATI
mount/calibration, sensor serial mapping, camera exposure, free disk, controller
deadman behavior, and a reachable physical emergency stop. First run
record-only, then a low-speed single episode with a spotter.

## Commands

Use the existing environment that contains OpenCV, HDF5, PyArrow, LeRobot,
OpenPI, JAX, and Torch:

```bash
OPENPI_PY=/home/suhang/datasets/openpi_env/openpi-venv/bin/python
cd /home/suhang/projects/droid
```

Ten-second no-robot dry run:

```bash
$OPENPI_PY tools/capture_fabric_droid.py /tmp/fabric_droid_smoke \
  --duration-sec 10 --destination-tray target_tray \
  --swatch-uid swatch_001 --split train --episode-id episode_smoke \
  --dry-run --robot-disabled
```

Real sensors, no robot connection:

```bash
$OPENPI_PY tools/capture_fabric_droid.py /data/fabric_droid/session_001 \
  --duration-sec 10 --destination-tray target_tray \
  --swatch-uid swatch_001 --split train --record-only --robot-disabled \
  --d435-serial D435_SERIAL --wrist-source /dev/video0 \
  --wrist-serial WRIST_SERIAL --gelsight-source /dev/video1 \
  --gelsight-serial GELSIGHT_SERIAL --ati-serial ATI_SERIAL \
  --ati-endpoint tcp://192.168.1.20:5555 --interactive-events
```

Guarded real DROID teleoperation (never run automatically):

```bash
$OPENPI_PY tools/run_fabric_droid_teleop.py \
  --enable-robot --preflight-confirmed \
  --destination-tray target_tray --swatch-uid swatch_001 --split train \
  --session-id session_001 --operator OPERATOR \
  --d435-serial D435_SERIAL --wrist-source /dev/video0 \
  --wrist-serial WRIST_SERIAL --gelsight-source /dev/video1 \
  --gelsight-serial GELSIGHT_SERIAL --ati-serial ATI_SERIAL
```

This guarded launcher still expects the base DROID camera/controller
configuration required by its GUI. Validate that configuration before the
pilot; it was not hardware-tested in this run.

Validation, events, and segmentation:

```bash
$OPENPI_PY tools/validate_fabric_droid_episode.py /path/to/episode
$OPENPI_PY tools/validate_fabric_droid_session.py /path/to/session
$OPENPI_PY tools/detect_fabric_droid_events.py /path/to/episode \
  --output /path/to/episode/tactile/automatic_event_candidates.json
$OPENPI_PY tools/segment_fabric_droid_episode.py /path/to/episode
```

Training export accepts the canonical target-tray task without tray pairing and
still forbids swatch split leakage:

```bash
HF_LEROBOT_HOME=/data/fabric_droid/lerobot
$OPENPI_PY tools/convert_fabric_droid_to_lerobot.py \
  /data/fabric_droid/session_001 \
  "$HF_LEROBOT_HOME/local/fabric_droid_v1" \
  --repo-id local/fabric_droid_v1 --include-validation

# Single-episode smoke:
$OPENPI_PY tools/convert_fabric_droid_to_lerobot.py \
  /tmp/fabric_droid_smoke /tmp/fabric_droid_lerobot \
  --repo-id local/fabric_droid_smoke
```

Batch and optional forward smoke:

```bash
PYTHONPATH=/home/suhang/projects/openpi/src \
  $OPENPI_PY tools/smoke_pi05_droid_batch.py \
  "$HF_LEROBOT_HOME/local/fabric_droid_v1" \
  --repo-id local/fabric_droid_v1

# Only when a local checkpoint has params/, assets/, and JAX sees a GPU:
PYTHONPATH=/home/suhang/projects/openpi/src \
  $OPENPI_PY tools/smoke_pi05_droid_batch.py \
  "$HF_LEROBOT_HOME/local/fabric_droid_v1" \
  --repo-id local/fabric_droid_v1 \
  --checkpoint-dir /data/checkpoints/pi05_droid_step
```

Future fine-tuning, with all OpenPI outputs redirected outside OpenPI:

```bash
cd /home/suhang/projects/openpi
HF_LEROBOT_HOME=/data/fabric_droid/lerobot \
  /home/suhang/datasets/openpi_env/openpi-venv/bin/python scripts/train.py \
  pi05_droid_finetune \
  --exp-name fabric_droid_v1 \
  --data.repo-id local/fabric_droid_v1 \
  --checkpoint-base-dir /data/fabric_droid/openpi_checkpoints \
  --assets-base-dir /data/fabric_droid/openpi_assets
```

This retains the official `pi05_droid` initialization and normalization assets.
No training was started.

Fabric-Omni read-only verification and Stage-3 cache:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/fabric_droid_pycache \
  $OPENPI_PY tools/verify_fabric_omni_bridge.py --load-checkpoint

PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/fabric_droid_pycache \
  $OPENPI_PY tools/cache_fabric_physical_state.py \
  /path/to/episode /data/fabric_droid/physical_cache --device cuda
```

The cache records checkpoint/config hashes. `softness_score` is explicitly the
flattening/compression response proxy, not a true softness label or classifier.
Heldout data is never used to fit normalization; the cache generator performs
no fitting.

## Verification completed

- Ruff check: passed for all new modules/tools/tests and the targeted DROID
  files. Pre-existing unrelated findings elsewhere in
  `droid/trajectory_utils/misc.py` were not rewritten; that whole legacy file
  was covered by the successful Python compile check.
- Python compile check: passed.
- Pytest: 12 passed.
- Ten-second synthetic episode:
  - robot/action: 150 frames, finite, monotonic, approximately 15 Hz;
  - external and wrist videos: 150 frames each, first/last decodable;
  - GelSight: 300 frames, first/last decodable, approximately 30 Hz;
  - ATI: 5,000 contiguous raw samples, 499.995 Hz, no decimation;
  - full stream overlap: 9.869 seconds;
  - required events and atomic COMPLETE marker present;
  - episode validator PASS.
- Session report, automatic event candidates, and valid Segment A/B manifest
  were generated.
- LeRobot conversion: one smoke episode, 150 frames. First and last frames
  loaded successfully.
- OpenPI batch: state `[8]`, actions `[16,8]`, base/left-wrist images present,
  right-wrist mask false, prompt retained.
- Final-170 checkpoint loaded on CPU: 278,161,925 initialized parameters,
  all frozen, four expected disabled lazy-RGB compatibility keys missing,
  zero unexpected keys.
- One real Final-170 CPU forward completed and wrote a 768-D physical cache
  under `/tmp`; no Fabric-Omni output was created.

## Not verified and remaining risks

- No D435/UVC device node was visible, `pyrealsense2` was unavailable in the
  OpenPI environment, and no ATI ZMQ sample arrived. Therefore the required
  real-sensor 10-second record-only smoke remains blocked by hardware/runtime
  exposure.
- NVIDIA driver access failed. The local `pi05_droid` checkpoint and
  normalization assets were absent. π0.5 forward and all training remain
  unverified and were not attempted.
- No Franka movement was commanded. DROID-to-FR3 networking, real controller
  semantics, estop reporting, calibrated workspace bounds, and the guarded
  teleoperation launcher require the physical pilot checklist.
- The OpenPI environment has a newer OpenCV whose `cv2.aruco` API cannot import
  the legacy DROID calibration module (`Dictionary_get` is absent). Run real
  teleoperation in DROID's pinned laptop Docker environment and install the
  `fabric-droid` optional dependencies there; do not use the OpenPI environment
  for robot control.
- The existing DROID GUI normally expects its configured camera set. Confirm
  how the site's D435/wrist streams coexist with or replace that base camera
  setup before enabling robot control.
- Disk headroom is low (about 15 GiB on the project filesystem and 1.6 GiB on
  the datasets filesystem at audit). Choose a larger output volume before
  collecting a real session.
- Real camera FPS/exposure behavior, ATI transport payload, force thresholds,
  `T_ati_to_gripper`, and the gripper-hold callback must be verified on the
  exact installed hardware.
