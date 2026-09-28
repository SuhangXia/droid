# Fabric-DROID Dataset Curator — audit and implementation report

Date: 2026-07-28  
Implementation: `/home/suhang/projects/droid/fabric_droid_curator`  
Raw root audited: `/home/suhang/datasets2/frabric_pi`  
Derived root: `/home/suhang/projects/droid/curation`

## 1. Repository audit

| Component | Resolved path | Git state |
|---|---|---|
| DROID | `/home/suhang/projects/droid` | `main`, remote `https://github.com/droid-dataset/droid`; existing local collection changes and untracked Fabric-DROID work were preserved |
| OpenPI | `/home/suhang/projects/openpi` | clean `main`, remote `https://github.com/Physical-Intelligence/openpi.git`, HEAD `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| Fabric-Omni | `/home/suhang/projects/fabric-omni` | no independent `.git`; Git resolves upward to `/home/suhang/projects`, an unrelated dirty parent worktree, so it was treated as read-only reference |
| ATI/GelSight reference | `/home/suhang/projects/test_code/2/octopi-1.5` | read-only reference |

DROID HEAD is `33ae6a67274f36d2e29525b86f23a56616ef43a7`.
The active acquisition entry points are
`fabric_droid/capture/recorder.py`,
`fabric_droid/capture/robot_trajectory.py`,
`fabric_droid/sensors/ati.py`,
`fabric_droid/sensors/camera.py`, and
`fabric_droid/ui/app.py`.

## 2. Episode layout and inventory

The raw root has 77 complete episode directories and one non-episode
`checkpoints/` directory. Other discovered trees were intentionally excluded:

- `/home/suhang/datasets2_repaired`: repaired copies;
- `/home/suhang/datasets2.pre_mount_leftovers_20260728`: incomplete captures;
- project output/in-progress directories;
- desktop Trash copies.

An actual episode contains:

```text
episode_*/
├── COMPLETE.json
├── device_assignment.json
├── metadata_episode_*.json
├── robot_telemetry.npz
├── trajectory.h5
├── recordings/
│   ├── MP4/
│   │   ├── exterior_image_1_left.mp4
│   │   ├── exterior_image_2_left.mp4  # declared duplicate
│   │   └── wrist_image_left.mp4
│   └── timestamps/
│       ├── exterior_image_1_left.npz
│       ├── exterior_image_2_left.npz
│       └── wrist_image_left.npz
└── tactile/
    ├── ati_raw.parquet
    ├── calibration.json
    ├── capture_report.json
    ├── events.json
    ├── gelsight_left.mp4
    ├── gelsight_left_timestamps.npy
    ├── gelsight_left_frame_metadata.npz
    └── ...
```

All 77 have `COMPLETE.json`, HDF5, the target videos, timestamp sidecars, and
metadata. All metadata currently say `success=true`.

## 3. Actual data schema

### Robot HDF5

`trajectory.h5` uses host monotonic nanoseconds and has these important fields:

| Field | Per-frame shape | dtype | Meaning |
|---|---:|---|---|
| `observation/robot_state/joint_positions` | `[7]` | float64 | measured joints |
| `observation/robot_state/joint_velocities` | `[7]` | float64 | measured joint velocity |
| `observation/robot_state/cartesian_position` | `[6]` | float64 | measured EEF position + orientation representation |
| `observation/robot_state/gripper_position` | scalar | float64 | measured gripper position |
| `action/joint_velocity` | `[7]` | float64 | stored policy action; collection attr identifies measured Franka joint velocity source |
| `action/gripper_position` | scalar | float64 | commanded absolute gripper target |
| `action/cartesian_position` | `[6]` | float64 | target EEF pose |
| `observation/timestamp/monotonic_ns` | scalar | int64 | shared host monotonic time |
| `observation/timestamp/robot_timestamp_seconds/nanos` | scalar | int64 | robot clock components |

EEF pose is saved. EEF linear/angular velocity is not a direct HDF5 dataset;
the curator derives it from pose and monotonic timestamps. Joint state,
measured joint velocity, measured gripper position, and commanded gripper
position are all present.

Across the 77 episodes:

- 41,814 policy frames;
- 127–880 frames per episode;
- 8.412–58.687 seconds;
- measured rate 14.959–14.978 Hz, median 14.975 Hz;
- all robot timestamps strictly monotonic;
- no state/action NaN or Inf was found.

### Video and timestamps

| Stream | Encoding | Resolution | timestamp sidecar | observed rate |
|---|---|---:|---|---:|
| D435 external | MPEG-4 Part 2, yuv420p | 640×480 | NPZ `timestamp_monotonic_ns`, receive/device timestamps, source frame index | 23.871–29.987 Hz |
| wrist/fisheye | MPEG-4 Part 2, yuv420p | 640×480 | same NPZ schema | 24.077–29.983 Hz |
| GelSight | MPEG-4 Part 2, yuv420p | 640×480 | NPY monotonic ns plus metadata NPZ | 18.624–18.761 Hz |

All 77 episodes' three target videos decode at the first and last frame.
Counts match timestamp sidecars: 83,834 external frames, 83,821 wrist frames,
and 55,636 GelSight frames.

The D435/wrist MP4 containers remain labeled 30 fps even when the measured
timestamp rate falls to 24–29 Hz. This is a synchronization warning, not a
corrupt-video hard failure. The UI maps shared monotonic timestamp to nearest
source frame and then to proxy media time; it never assumes frame indices are
synchronous across modalities.

### ATI

`ati_raw.parquet` columns are:

```text
timestamp_monotonic_ns int64
timestamp_wall_ns      int64
sample_index           int64
Fx,Fy,Fz,Tx,Ty,Tz      float64
device_status          string
device_timestamp_ns    int64
```

There are 1,431,461 samples. Non-empty episodes measure approximately 500 Hz
and use contiguous sample indices. Three completed episodes have an empty ATI
table: `episode_20260727_223727`, `episode_20260727_224035`, and
`episode_20260727_224934`.

### Metadata

Every current raw episode contains the same collected categorical values:
`swatch_001`, `train`, `ready`, `source_slot_1`, and `target_tray`. Metadata
does not yet contain curator review, action branch, or camera-session labels.
Legacy project outputs contain `destination_tray=green_left`, which maps only
to `legacy_green_left`; the schema also supports `legacy_white_right`.
Neither legacy label is converted to a material property.

Every stream has a host monotonic timestamp. Existing `events.json` files do
not contain `probe_complete` or `release_time`; automatic proposals therefore
remain proposals until human review.

## 4. Full index and QC result

The complete raw root was indexed read-only:

```text
discovered=77 indexed=76 unchanged=1 failed=0
```

The derived index occupies about 66 MiB without eagerly proxying every video.
It contains 77 signal caches, thumbnails, event proposals, QC, and DB rows.
Browser-compatible H.264 proxies are incremental/on-demand.

Current automatic queue:

- Green: 16;
- Yellow: 57;
- Red: 4;
- hard failures: three empty ATI streams and one episode with negligible
  gripper variation (`episode_20260727_195932`);
- common warnings: encoded fps differs from timestamp rate, large camera frame
  gaps, and automatically estimated hold shorter than 0.25 seconds.

The metadata-derived camera ID is
`session_20260726__102422075665`. Comparing all 77 reference thumbnails
classifies the aggregate as `major_shift`; this is a review/robustness flag,
not a deletion decision. Appearance and robot pose can confound the simple
image/homography detector.

## 5. Database and annotation schema

SQLite has:

```text
episodes
sensor_streams
event_proposals
annotations
segments
reviews
exports
camera_sessions
qc_results
audit_logs
schema_migrations
```

Each annotation save appends a version. JSON snapshots are also written as
`annotations/<episode>.vNNN.json`. Audit rows keep old/new values, reviewer,
reason, timestamp, and annotation version. Undo/Redo creates another immutable
version by restoring a selected earlier payload.

The annotation model separates:

- episode/review fields;
- `action_branch`: remove, leave, two legacy values, or unknown;
- timestamped task events;
- material fields: summer suitability, softness, thickness, breathability,
  semantic confidence, and label source.

An accepted or verified annotation cannot have an unknown action branch.
Action branch never fills material semantics.

## 6. Automatic events and QC

The detector emits timestamp, confidence, evidence dictionary, and
`fabric-curator-fusion-v1` for each proposal:

1. `motion_start`: sustained joint/EEF movement above robust baseline.
2. `contact_start`: fused gripper closing, GelSight contact/image delta, and
   ATI baseline-relative force. No ATI-only threshold is used.
3. `stable_grasp`: contact persistence, low gripper velocity, lower ATI
   derivative, and stable GelSight contact.
4. `lift_start`: post-grasp sustained joint/EEF motion.
5. `detach_complete`: post-lift force peak/drop constrained by task order;
   explicitly marked as requiring manual confirmation.
6. `release_start`: gripper opening evidence.
7. `release_complete`: measured open state plus reduced tactile/force evidence.
8. `retreat_complete`: final sustained low-motion interval.

Low-evidence fallbacks have low confidence and retain their evidence. On the
real integration episode, `stable_grasp` was high confidence while
`lift_start` was only 0.08, demonstrating that the UI surfaces uncertainty
instead of claiming accurate automatic cuts.

QC persists the nine requested scores and Red/Yellow/Green severity.
Calibrated normal force is not derived because the current metadata says
`calibration_id=uncalibrated`.

## 7. Segment definitions

Manual review uses one `branch_point`, initially seeded from the automatic
`stable_grasp` proposal. It makes exactly one non-overlapping cut:

- `grasp_probe`: episode start to `branch_point`;
- `remove_to_basket` or `leave_on_rack`: `branch_point` to episode end;
- `recovery`: the complete episode when explicitly marked as recovery.

Prompts come only from versioned fixed templates. Manifests preserve the cut
as `split_point_ns`, plus the contact/stable timestamps and K=1 tactile window.
Older annotations without `branch_point` are upgraded in memory from
`stable_grasp` and persist it on the next save.

## 8. UI

The React UI provides:

- shared monotonic playhead over external, wrist, and GelSight H.264 proxies;
- actual source timestamp and nearest frame/policy index display;
- previous/next policy frame, playback rates, and cut-point navigation;
- one draggable, unlabelled cut marker on the timeline and signal charts;
- robot, gripper, ATI, EEF, and GelSight ECharts with zoom;
- automatic versioned saving and Undo/Redo;
- two large, mutually highlighted Remove/Leave buttons with immediate
  versioned persistence, plus Accept/Reject/Recovery/Verified and Pass 1 fields;
- segment-only preview with start/end band, prompt, warnings, and source;
- QC priority and score display;
- dashboard totals, balance matrices, distributions, split counts, and camera
  session reference.

Keyboard shortcuts are documented in the package README.

## 9. Export and downstream compatibility

An immutable export includes:

```text
manifest.parquet
annotation_snapshot.json
instruction_templates.json
split_manifest.json
segment_stats.json
rejected_episodes.json
recovery_episodes.json
leakage_audit.json
sensor_qc_report.json
camera_session_report.json
export_config.yaml
```

Split assignment is by `swatch_uid`; duplicate assignments abort export.
Derived segments inherit the episode split. Heldout is excluded unless
explicitly requested, and no normalization is fitted by the curator.

The smoke converter slices only from the manifest timestamps and builds:

```text
state  = 7D joint position + 1D measured gripper
action = 7D joint velocity + 1D commanded gripper target
```

It also timestamp-aligns external/wrist images, checks lengths and finite
values, retains the fixed prompt, and constructs a π0.5-style state/action
horizon/image batch. It does not launch training.

Fabric-Omni integration is intentionally an interface boundary: use the
manifest's tactile window to load GelSight/ATI/gripper state, then call
`fabric_encoder.encode_policy_state(...)` outside the curator. Heldout rows
must be rejected by any training-cache job.

## 10. Verification

Passed:

- Ruff on all curator Python and curator tests;
- 10 pytest unit/integration cases;
- real episode read-only index, all three video decodes, HDF5/ATI read, event
  detection, annotation persistence, two segment types, immutable Parquet
  manifest, leakage report, and two-segment LeRobot/π0.5 batch smoke;
- byte size and modification time of every file in the real source episode
  unchanged before/after integration;
- TypeScript and Vite production build;
- Playwright with system Chrome: page load, three video Range streams, shared
  timeline, one cut marker, three charts, policy-frame step, immediate branch
  selection, marker drag, autosave, refresh persistence, segment preview, and
  play/pause.

The broader pre-existing Fabric-DROID test selection had 43 passes and one
skip; two unrelated tests could not import optional `PySide6` and `torch` in
the curator-only environment. These are environment dependency failures, not
curator test regressions.

## 11. Not verified and known risks

- No automatic event timestamp has been accepted as human ground truth.
- No episode is claimed human-Verified by this implementation run.
- The three empty ATI episodes cannot satisfy tactile completeness.
- The simple camera drift estimate can confuse scene/pose changes with camera
  motion; major shifts require human review.
- Browser proxies are H.264 derived previews and do not replace raw MP4.
- Full eager proxy generation for all 77 episodes was not run; proxies are
  tested and generated on demand.
- Formal LeRobot dataset materialization, OpenPI model forward/training,
  π0.5 training, Fabric-Omni/Qwen execution, and normalization fitting were
  not run.
- No heldout data was used for training or normalization; the current audited
  raw root labels every item `train`, so meaningful heldout swatches still
  need to be configured after correct `swatch_uid` labeling.
- No robot-control or robot-movement interface was called.

## 12. Recommended next steps

1. Correct the 77 episodes' `swatch_uid` and define train/validation/heldout by
   physical swatch before any training export.
2. Run Pass 1, starting with four Red and 57 Yellow episodes.
3. For each episode, set the one cut point, choose Remove or Leave, and mark
   only reviewed rows Verified.
4. Review the camera major-shift subset and split true physical tripod changes
   into explicit camera session IDs.
5. Export a new immutable version and run the manifest smoke command.
6. Add a manifest consumer in OpenPI and a tactile-window loader in
   Fabric-Omni; enforce heldout exclusion again at both cache boundaries.
