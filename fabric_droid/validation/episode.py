"""Validation for one independently readable Fabric-DROID episode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fabric_droid.sync.clock import analyze_timestamps


def _video_report(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        return {"path": str(path), "decodable": False, "error": f"opencv unavailable: {exc}"}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"path": str(path), "decodable": False, "frame_count": 0}
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    first_ok, _ = capture.read()
    last_ok = False
    if count > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
        last_ok, _ = capture.read()
    capture.release()
    return {
        "path": str(path),
        "decodable": bool(first_ok and last_ok),
        "frame_count": count,
        "fps": fps,
    }


def _nearest_max_delta_ns(reference: np.ndarray, query: np.ndarray) -> int | None:
    if reference.size == 0 or query.size == 0:
        return None
    indices = np.searchsorted(reference, query)
    right = np.abs(reference[np.minimum(indices, reference.size - 1)] - query)
    left = np.abs(reference[np.maximum(indices - 1, 0)] - query)
    return int(np.minimum(left, right).max())


def validate_episode(
    episode_dir: Path,
    *,
    minimum_duration_sec: float = 1.0,
    require_success: bool = True,
    require_segmentation_events: bool = False,
) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    failures: list[str] = []
    warnings: list[str] = []
    required = [
        episode_dir / "trajectory.h5",
        episode_dir / "tactile" / "gelsight_left.mp4",
        episode_dir / "tactile" / "gelsight_left_timestamps.npy",
        episode_dir / "tactile" / "ati_raw.parquet",
        episode_dir / "tactile" / "events.json",
        episode_dir / "tactile" / "calibration.json",
        episode_dir / "tactile" / "capture_report.json",
        episode_dir / "COMPLETE.json",
    ]
    for path in required:
        if not path.is_file():
            failures.append(f"missing required file: {path.relative_to(episode_dir)}")
    metadata_paths = list(episode_dir.glob("metadata_*.json"))
    if len(metadata_paths) != 1:
        failures.append(f"expected exactly one metadata_*.json, found {len(metadata_paths)}")
        metadata: dict[str, Any] = {}
    else:
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))

    robot_timestamps = np.asarray([], dtype=np.int64)
    robot_length = action_length = 0
    gripper_range = [None, None]
    state_action_finite = False
    if (episode_dir / "trajectory.h5").is_file():
        try:
            import h5py

            with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
                joints = np.asarray(handle["observation/robot_state/joint_positions"])
                gripper = np.asarray(handle["observation/robot_state/gripper_position"])
                actions = np.asarray(handle["action/joint_velocity"])
                commanded_gripper = np.asarray(handle["action/gripper_position"])
                robot_timestamps = np.asarray(handle["observation/timestamp/monotonic_ns"], dtype=np.int64)
                robot_length = int(joints.shape[0])
                action_length = int(actions.shape[0])
                state_action_finite = bool(
                    np.isfinite(joints).all()
                    and np.isfinite(gripper).all()
                    and np.isfinite(actions).all()
                    and np.isfinite(commanded_gripper).all()
                )
                if gripper.size:
                    gripper_range = [float(gripper.min()), float(gripper.max())]
                h5_success = bool(handle.attrs.get("success", False))
        except Exception as exc:
            failures.append(f"trajectory.h5 unreadable or incompatible: {exc}")
            h5_success = False
        else:
            if robot_length != action_length:
                failures.append(f"robot/action length mismatch: {robot_length} vs {action_length}")
            if not state_action_finite:
                failures.append("robot state or action contains NaN/Inf")
            if not analyze_timestamps(robot_timestamps).monotonic:
                failures.append("robot timestamps are not strictly monotonic")
            if require_success and not h5_success:
                failures.append("trajectory is not marked successful")

    camera_reports: dict[str, Any] = {}
    camera_timestamps: dict[str, np.ndarray] = {}
    for name in ("exterior_image_1_left", "wrist_image_left"):
        video_path = episode_dir / "recordings" / "MP4" / f"{name}.mp4"
        timestamp_path = episode_dir / "recordings" / "timestamps" / f"{name}.npz"
        camera_reports[name] = _video_report(video_path)
        if not camera_reports[name].get("decodable"):
            failures.append(f"{name} video is missing or corrupt")
        if timestamp_path.is_file():
            with np.load(timestamp_path) as payload:
                camera_timestamps[name] = np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)
            clock = analyze_timestamps(camera_timestamps[name])
            camera_reports[name]["clock"] = clock.to_dict()
            if not clock.monotonic:
                failures.append(f"{name} timestamps are not strictly monotonic")
            if camera_reports[name].get("frame_count") != clock.count:
                failures.append(f"{name} video/timestamp frame count mismatch")
            encoded_fps = float(camera_reports[name].get("fps", 0.0))
            if (
                clock.count > 1
                and encoded_fps > 0
                and abs(encoded_fps - clock.measured_hz) > 0.5
            ):
                failures.append(
                    f"{name} encoded fps {encoded_fps:.3f} does not match "
                    f"timestamp rate {clock.measured_hz:.3f}"
                )
            if (
                clock.measured_hz > 0
                and clock.max_gap_ms
                > max(100.0, 2500.0 / clock.measured_hz)
            ):
                warnings.append(
                    f"{name} has a large frame gap: {clock.max_gap_ms:.3f} ms"
                )
        else:
            failures.append(f"missing {name} timestamp sidecar")

    gelsight_path = episode_dir / "tactile" / "gelsight_left_timestamps.npy"
    gelsight_timestamps = np.load(gelsight_path) if gelsight_path.is_file() else np.asarray([], dtype=np.int64)
    gelsight_video = _video_report(episode_dir / "tactile" / "gelsight_left.mp4")
    gelsight_clock = analyze_timestamps(gelsight_timestamps)
    if not gelsight_video.get("decodable"):
        failures.append("GelSight video is missing or corrupt")
    if gelsight_video.get("frame_count") != gelsight_clock.count:
        failures.append("GelSight video/timestamp frame count mismatch")
    if not gelsight_clock.monotonic:
        failures.append("GelSight timestamps are not strictly monotonic")
    gelsight_encoded_fps = float(gelsight_video.get("fps", 0.0))
    if (
        gelsight_clock.count > 1
        and gelsight_encoded_fps > 0
        and abs(gelsight_encoded_fps - gelsight_clock.measured_hz) > 0.5
    ):
        failures.append(
            f"GelSight encoded fps {gelsight_encoded_fps:.3f} does not match "
            f"timestamp rate {gelsight_clock.measured_hz:.3f}"
        )
    if (
        gelsight_clock.measured_hz > 0
        and gelsight_clock.max_gap_ms
        > max(100.0, 2500.0 / gelsight_clock.measured_hz)
    ):
        warnings.append(
            f"GelSight has a large frame gap: {gelsight_clock.max_gap_ms:.3f} ms"
        )

    ati_timestamps = np.asarray([], dtype=np.int64)
    ati_ranges: dict[str, list[float]] = {}
    ati_path = episode_dir / "tactile" / "ati_raw.parquet"
    if ati_path.is_file():
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(ati_path)
            ati_timestamps = np.asarray(table["timestamp_monotonic_ns"], dtype=np.int64)
            for name in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"):
                values = np.asarray(table[name], dtype=np.float64)
                ati_ranges[name] = [float(values.min()), float(values.max())] if values.size else [0.0, 0.0]
            indices = np.asarray(table["sample_index"], dtype=np.int64)
            if indices.size > 1 and not np.all(np.diff(indices) == 1):
                failures.append("ATI sample_index is not contiguous; possible dropped or reordered samples")
            wrench = np.column_stack(
                [np.asarray(table[name], dtype=np.float64) for name in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")]
            )
            if not wrench.size:
                failures.append("ATI contains no force/torque samples")
            elif not np.isfinite(wrench).all():
                failures.append("ATI force/torque contains NaN/Inf")
            elif not np.any(np.abs(wrench) > 1e-9):
                failures.append("ATI force/torque samples are all zero")
        except Exception as exc:
            failures.append(f"ATI Parquet unreadable: {exc}")
    ati_clock = analyze_timestamps(ati_timestamps)
    if not ati_clock.monotonic:
        failures.append("ATI timestamps are not strictly monotonic")
    if ati_clock.count and not 450.0 <= ati_clock.measured_hz <= 550.0:
        failures.append(f"ATI rate {ati_clock.measured_hz:.2f} Hz is outside [450, 550]")

    events_path = episode_dir / "tactile" / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8")).get("events", []) if events_path.is_file() else []
    event_names = {event.get("name") for event in events}
    for required_event in ("probe_complete", "release_time"):
        if required_event not in event_names:
            target = failures if require_segmentation_events else warnings
            target.append(f"missing event: {required_event}")
    event_times = np.asarray([event.get("timestamp_monotonic_ns", 0) for event in events], dtype=np.int64)
    if event_times.size and np.any(np.diff(event_times) < 0):
        failures.append("events are not chronological")

    all_streams = [robot_timestamps, gelsight_timestamps, ati_timestamps, *camera_timestamps.values()]
    nonempty = [values for values in all_streams if values.size]
    overlap_sec = 0.0
    if nonempty:
        overlap_start = max(int(values[0]) for values in nonempty)
        overlap_end = min(int(values[-1]) for values in nonempty)
        overlap_sec = max(0.0, (overlap_end - overlap_start) / 1e9)
        if overlap_sec < minimum_duration_sec * 0.8:
            failures.append(f"stream overlap is only {overlap_sec:.3f}s")
    duration_sec = analyze_timestamps(robot_timestamps).duration_sec
    if duration_sec < minimum_duration_sec:
        failures.append(f"episode duration {duration_sec:.3f}s is below {minimum_duration_sec:.3f}s")
    max_sync_delta_ns = max(
        (
            value
            for values in camera_timestamps.values()
            if (value := _nearest_max_delta_ns(robot_timestamps, values)) is not None
        ),
        default=None,
    )
    if gripper_range[0] is not None and gripper_range[1] - gripper_range[0] < 1e-4:
        warnings.append("gripper measured position has negligible variation")
    if require_success and not metadata.get("success", False):
        failures.append("metadata success is false")

    report = {
        "episode_dir": str(episode_dir),
        "pass": not failures,
        "failures": failures,
        "warnings": warnings,
        "complete": (episode_dir / "COMPLETE.json").is_file(),
        "success": bool(metadata.get("success", False)),
        "failure_reason": metadata.get("failure_reason", ""),
        "metadata": metadata,
        "duration_sec": duration_sec,
        "robot_length": robot_length,
        "action_length": action_length,
        "state_action_finite": state_action_finite,
        "robot_clock": analyze_timestamps(robot_timestamps).to_dict(),
        "cameras": camera_reports,
        "gelsight": {"video": gelsight_video, "clock": gelsight_clock.to_dict()},
        "ati": {"clock": ati_clock.to_dict(), "raw_ranges": ati_ranges},
        "overlap_sec": overlap_sec,
        "maximum_robot_camera_sync_delta_ms": None if max_sync_delta_ns is None else max_sync_delta_ns / 1e6,
        "gripper_range": gripper_range,
        "required_events": {name: name in event_names for name in ("probe_complete", "release_time")},
    }
    return report
