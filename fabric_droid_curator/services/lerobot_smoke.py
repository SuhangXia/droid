from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import yaml


def _decode_selected(video_path: Path, timestamp_path: Path, target_ns: np.ndarray, *, size: int = 224) -> np.ndarray:
    if timestamp_path.suffix == ".npy":
        frame_ns = np.asarray(np.load(timestamp_path), dtype=np.int64)
    else:
        with np.load(timestamp_path) as payload:
            frame_ns = np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)
    right = np.searchsorted(frame_ns, target_ns)
    right = np.clip(right, 0, max(0, frame_ns.size - 1))
    left = np.clip(right - 1, 0, max(0, frame_ns.size - 1))
    nearest = np.where(
        np.abs(frame_ns[right] - target_ns) < np.abs(frame_ns[left] - target_ns),
        right,
        left,
    )
    requested = {int(index): [] for index in np.unique(nearest)}
    for output_index, frame_index in enumerate(nearest):
        requested[int(frame_index)].append(output_index)
    images = np.empty((target_ns.size, size, size, 3), dtype=np.uint8)
    capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    while requested:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in requested:
            rgb = cv2.cvtColor(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            for output_index in requested.pop(frame_index):
                images[output_index] = rgb
        frame_index += 1
    capture.release()
    if requested:
        raise RuntimeError(f"could not decode requested frames {sorted(requested)[:5]} from {video_path}")
    return images


def smoke_manifest(manifest_path: Path, *, data_root: Path | None = None, max_segments: int = 2) -> dict[str, Any]:
    table = pq.read_table(manifest_path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError("manifest contains no segments")
    if data_root is None:
        config_path = manifest_path.parent / "export_config.yaml"
        if not config_path.is_file():
            raise ValueError("--data-root is required when export_config.yaml is unavailable")
        data_root = Path(yaml.safe_load(config_path.read_text(encoding="utf-8"))["data_root"])
    reports = []
    for row in rows[:max_segments]:
        episode_dir = data_root / row["source_episode_id"]
        with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
            timestamp_ns = np.asarray(handle["observation/timestamp/monotonic_ns"], dtype=np.int64)
            mask = (timestamp_ns >= row["start_ns"]) & (timestamp_ns <= row["end_ns"])
            indices = np.flatnonzero(mask)
            if not indices.size:
                raise ValueError(f"segment {row['segment_id']} has no policy frames")
            state = np.column_stack(
                [
                    np.asarray(handle["observation/robot_state/joint_positions"])[indices],
                    np.asarray(handle["observation/robot_state/gripper_position"])[indices],
                ]
            ).astype(np.float32)
            action = np.column_stack(
                [
                    np.asarray(handle["action/joint_velocity"])[indices],
                    np.asarray(handle["action/gripper_position"])[indices],
                ]
            ).astype(np.float32)
            selected_ns = timestamp_ns[indices]
        if state.shape != action.shape or state.shape[1] != 8:
            raise ValueError(f"state/action shape mismatch for {row['segment_id']}: {state.shape}, {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"non-finite policy values in {row['segment_id']}")
        external = _decode_selected(
            episode_dir / "recordings/MP4/exterior_image_1_left.mp4",
            episode_dir / "recordings/timestamps/exterior_image_1_left.npz",
            selected_ns,
        )
        wrist = _decode_selected(
            episode_dir / "recordings/MP4/wrist_image_left.mp4",
            episode_dir / "recordings/timestamps/wrist_image_left.npz",
            selected_ns,
        )
        horizon = min(16, action.shape[0])
        pi05_batch = {
            "observation/state": state[0],
            "actions": action[:horizon],
            "observation/image": external[0],
            "observation/wrist_image": wrist[0],
            "prompt": row["prompt"],
        }
        if pi05_batch["observation/state"].shape != (8,) or pi05_batch["actions"].shape[1] != 8:
            raise ValueError("pi05_droid smoke batch has invalid shape")
        reports.append(
            {
                "segment_id": row["segment_id"],
                "segment_type": row["segment_type"],
                "policy_frame_count": int(state.shape[0]),
                "state_shape": list(state.shape),
                "action_shape": list(action.shape),
                "external_image_shape": list(external.shape),
                "wrist_image_shape": list(wrist.shape),
                "prompt": row["prompt"],
                "requested_range_ns": [row["start_ns"], row["end_ns"]],
                "actual_range_ns": [int(selected_ns[0]), int(selected_ns[-1])],
                "pi05_batch": {
                    "state_shape": list(pi05_batch["observation/state"].shape),
                    "actions_shape": list(pi05_batch["actions"].shape),
                    "image_shape": list(pi05_batch["observation/image"].shape),
                    "wrist_image_shape": list(pi05_batch["observation/wrist_image"].shape),
                },
                "finite": True,
            }
        )
    return {
        "manifest": str(manifest_path),
        "smoke_only": True,
        "formal_training_started": False,
        "normalization_fitted": False,
        "segments": reports,
    }
