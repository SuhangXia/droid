#!/usr/bin/env python3
"""Materialize a Fabric-DROID curator manifest as a LeRobot dataset.

Each active curator segment becomes one LeRobot episode.  This is important for
the Fabric-DROID labels: the raw metadata has one generic task instruction,
while the curator manifest contains the action-specific prompts and cut times.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to decode {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            frames.append(frame[..., ::-1].copy())
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return frames


def _timestamps(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)


def _nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_ns, target_ns)
    right = np.minimum(right, source_ns.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(source_ns[left] - target_ns) <= np.abs(source_ns[right] - target_ns)
    return np.where(choose_left, left, right)


def _load_episode(episode_dir: Path) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[np.ndarray]]:
    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        arrays = {
            "timestamps": np.asarray(handle["observation/timestamp/monotonic_ns"], dtype=np.int64),
            "joint_position": np.asarray(handle["observation/robot_state/joint_positions"], dtype=np.float32),
            "gripper_position": np.asarray(handle["observation/robot_state/gripper_position"], dtype=np.float32),
            "joint_velocity": np.asarray(handle["action/joint_velocity"], dtype=np.float32),
            "commanded_gripper_position": np.asarray(handle["action/gripper_position"], dtype=np.float32),
        }
    recording = episode_dir / "recordings"
    exterior = _video_frames(recording / "MP4/exterior_image_1_left.mp4")
    wrist = _video_frames(recording / "MP4/wrist_image_left.mp4")
    exterior_ns = _timestamps(recording / "timestamps/exterior_image_1_left.npz")
    wrist_ns = _timestamps(recording / "timestamps/wrist_image_left.npz")
    arrays["exterior_indices"] = _nearest_indices(exterior_ns, arrays["timestamps"])
    arrays["wrist_indices"] = _nearest_indices(wrist_ns, arrays["timestamps"])
    lengths = {key: value.shape[0] for key, value in arrays.items() if value.ndim > 0}
    if len(set(lengths[key] for key in ("timestamps", "joint_position", "gripper_position", "joint_velocity", "commanded_gripper_position"))) != 1:
        raise ValueError(f"trajectory arrays have inconsistent lengths: {lengths}")
    return arrays, exterior, wrist


def convert_manifest(manifest: Path, raw_root: Path, output_root: Path, repo_id: str) -> dict[str, object]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    rows = pq.read_table(manifest).to_pylist()
    if not rows:
        raise ValueError("manifest contains no segments")
    by_episode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "train":
            raise ValueError(f"non-train segment found: {row['segment_id']}")
        if row["dataset_decision"] != "keep":
            raise ValueError(f"non-kept segment found: {row['segment_id']}")
        by_episode[str(row["source_episode_id"])].append(row)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type="franka",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
            "timestamp_monotonic_ns": {"dtype": "int64", "shape": (1,), "names": ["timestamp_monotonic_ns"]},
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )

    episode_count = 0
    frame_count = 0
    segment_counts: dict[str, int] = defaultdict(int)
    for episode_id in sorted(by_episode):
        episode_dir = raw_root / episode_id
        arrays, exterior, wrist = _load_episode(episode_dir)
        timestamps = arrays["timestamps"]
        for row in sorted(by_episode[episode_id], key=lambda item: int(item["start_ns"])):
            mask = (timestamps >= int(row["start_ns"])) & (timestamps <= int(row["end_ns"]))
            indices = np.flatnonzero(mask)
            if indices.size < 16:
                raise ValueError(f"segment {row['segment_id']} has only {indices.size} policy frames")
            prompt = str(row["prompt"])
            for index in indices:
                exterior_image = exterior[int(arrays["exterior_indices"][index])]
                wrist_image = wrist[int(arrays["wrist_indices"][index])]
                action = np.concatenate(
                    [
                        arrays["joint_velocity"][index],
                        np.asarray([arrays["commanded_gripper_position"][index]], dtype=np.float32),
                    ]
                ).astype(np.float32)
                dataset.add_frame(
                    {
                        "exterior_image_1_left": exterior_image,
                        "exterior_image_2_left": exterior_image.copy(),
                        "wrist_image_left": wrist_image,
                        "joint_position": arrays["joint_position"][index],
                        "gripper_position": np.asarray([arrays["gripper_position"][index]], dtype=np.float32),
                        "actions": action,
                        "timestamp_monotonic_ns": np.asarray([timestamps[index]], dtype=np.int64),
                        "task": prompt,
                    }
                )
                frame_count += 1
            dataset.save_episode()
            episode_count += 1
            segment_counts[str(row["segment_type"])] += 1

    report = {
        "repo_id": repo_id,
        "output_root": str(output_root),
        "episode_count": episode_count,
        "frame_count": frame_count,
        "segment_counts": dict(sorted(segment_counts.items())),
        "source_episode_count": len(by_episode),
        "manifest": str(manifest),
    }
    (output_root / "curator_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    convert_manifest(args.manifest, args.raw_root, args.output_root, args.repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
