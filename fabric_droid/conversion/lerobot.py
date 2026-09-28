"""Fabric-DROID to LeRobot converter compatible with OpenPI pi05_droid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fabric_droid.validation.episode import validate_episode


def _read_video(path: Path) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to decode {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = frame[..., ::-1]
            if tuple(rgb.shape[:2]) != (180, 320):
                rgb = cv2.resize(rgb, (320, 180), interpolation=cv2.INTER_AREA)
            frames.append(rgb.copy())
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return frames


def _nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(source_ns, target_ns)
    right = np.minimum(indices, source_ns.size - 1)
    left = np.maximum(indices - 1, 0)
    choose_left = np.abs(source_ns[left] - target_ns) <= np.abs(source_ns[right] - target_ns)
    return np.where(choose_left, left, right)


def _load_episode(episode_dir: Path) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[np.ndarray], str]:
    import h5py

    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        arrays = {
            "timestamps": np.asarray(handle["observation/timestamp/monotonic_ns"], dtype=np.int64),
            "joint_position": np.asarray(handle["observation/robot_state/joint_positions"], dtype=np.float32),
            "gripper_position": np.asarray(handle["observation/robot_state/gripper_position"], dtype=np.float32),
            "joint_velocity": np.asarray(handle["action/joint_velocity"], dtype=np.float32),
            "commanded_gripper_position": np.asarray(handle["action/gripper_position"], dtype=np.float32),
        }
    exterior = _read_video(episode_dir / "recordings" / "MP4" / "exterior_image_1_left.mp4")
    wrist = _read_video(episode_dir / "recordings" / "MP4" / "wrist_image_left.mp4")
    with np.load(episode_dir / "recordings" / "timestamps" / "exterior_image_1_left.npz") as payload:
        exterior_ns = np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)
    with np.load(episode_dir / "recordings" / "timestamps" / "wrist_image_left.npz") as payload:
        wrist_ns = np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)
    arrays["exterior_indices"] = _nearest_indices(exterior_ns, arrays["timestamps"])
    arrays["wrist_indices"] = _nearest_indices(wrist_ns, arrays["timestamps"])
    metadata_path = next(iter(episode_dir.glob("metadata_*.json")))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return arrays, exterior, wrist, metadata["task_instruction"]


def convert_session(
    data_dir: Path,
    output_root: Path,
    repo_id: str,
    *,
    include_splits: tuple[str, ...] = ("train", "validation"),
    minimum_duration_sec: float = 1.0,
) -> dict[str, Any]:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot is required; run this command in the OpenPI environment") from exc
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty LeRobot root: {output_root}")
    episode_dirs = sorted({path.parent for path in data_dir.glob("**/trajectory.h5")})
    selected: list[Path] = []
    skipped: dict[str, str] = {}
    swatch_splits: dict[str, set[str]] = {}
    for episode_dir in episode_dirs:
        report = validate_episode(episode_dir, minimum_duration_sec=minimum_duration_sec)
        metadata = report["metadata"]
        swatch_splits.setdefault(str(metadata.get("swatch_uid")), set()).add(str(metadata.get("split")))
        if not report["pass"]:
            skipped[str(episode_dir)] = "; ".join(report["failures"])
        elif metadata.get("split") not in include_splits:
            skipped[str(episode_dir)] = f"split {metadata.get('split')} excluded"
        else:
            selected.append(episode_dir)
    leaked = {swatch: splits for swatch, splits in swatch_splits.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"swatch split leakage detected: {leaked}")
    if not selected:
        raise ValueError("no valid successful episodes selected")
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
            # OpenPI's current LeRobotDROIDDataConfig repacks this key, although
            # DroidInputs intentionally does not consume it. Duplicate camera 1.
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
            "timestamp_monotonic_ns": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["timestamp_monotonic_ns"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )
    total_frames = 0
    for episode_dir in selected:
        arrays, exterior, wrist, prompt = _load_episode(episode_dir)
        for index, timestamp in enumerate(arrays["timestamps"]):
            exterior_image = exterior[int(arrays["exterior_indices"][index])]
            wrist_image = wrist[int(arrays["wrist_indices"][index])]
            dataset.add_frame(
                {
                    "exterior_image_1_left": exterior_image,
                    "exterior_image_2_left": exterior_image.copy(),
                    "wrist_image_left": wrist_image,
                    "joint_position": arrays["joint_position"][index],
                    "gripper_position": np.asarray([arrays["gripper_position"][index]], dtype=np.float32),
                    "actions": np.concatenate(
                        [
                            arrays["joint_velocity"][index],
                            np.asarray([arrays["commanded_gripper_position"][index]], dtype=np.float32),
                        ]
                    ).astype(np.float32),
                    "timestamp_monotonic_ns": np.asarray([timestamp], dtype=np.int64),
                    "task": prompt,
                }
            )
            total_frames += 1
        dataset.save_episode()
    return {
        "repo_id": repo_id,
        "output_root": str(dataset.root),
        "episode_count": len(selected),
        "frame_count": total_frames,
        "selected_episodes": [str(path) for path in selected],
        "skipped_episodes": skipped,
        "action_space": "joint_velocity_7_plus_commanded_gripper_position_1",
        "heldout_included": False,
        "task_label_mode": "canonical_target_tray",
    }
