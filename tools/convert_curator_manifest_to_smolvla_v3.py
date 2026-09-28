#!/usr/bin/env python3
"""Build a SmolVLA-ready LeRobot v3 dataset from a Fabric-DROID export.

This intentionally consumes the immutable curator ``manifest.parquet`` rather
than the per-episode task metadata.  The latter was canonicalised for the old
Pi0.5 workflow and no longer distinguishes the two branches (remove versus
leave).  Each kept manifest segment becomes one LeRobot episode, with the
manifest prompt retained as its task string.

The source data is never modified.  For safety, this command refuses a target
directory that already exists; use ``--validate-only`` to inspect an existing
output instead of rebuilding it.

Example (run inside a LeRobot >= 0.4 environment):

    python tools/convert_curator_manifest_to_smolvla_v3.py \
      --manifest curation/exports/fabric_droid_v001/manifest.parquet \
      --raw-root /home/suhang/datasets2/frabric_pi \
      --output-root /home/suhang/datasets2/smolvla/fabric_droid_v001_v3 \
      --repo-id local/fabric_droid_v001_smolvla_v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq


FPS = 15
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EXTERIOR_KEY = "observation.images.exterior_1_left"
WRIST_KEY = "observation.images.wrist_left"
REQUIRED_MANIFEST_COLUMNS = {
    "segment_id",
    "source_episode_id",
    "segment_type",
    "action_branch",
    "prompt",
    "start_ns",
    "end_ns",
    "split",
    "dataset_decision",
}

# This is the immutable curator export used for the previous Pi0.5 run.  The
# lock prevents an accidental conversion of a raw-session manifest (which can
# contain rejected rows or canonicalised task labels) under the same run name.
V001_MANIFEST_SHA256 = "d6e031d61f6c3bc541ecb0c37fffbc86260f111891f38a2b44ada616656f5243"
V001_SEGMENT_COUNTS = {
    "grasp_probe": 126,
    "leave_on_rack": 62,
    "remove_to_basket": 64,
}
V001_ACTION_BRANCH_COUNTS = {"leave": 124, "remove": 128}
V001_SOURCE_EPISODE_COUNT = 126
V001_SEGMENT_COUNT = 252
V001_FRAME_COUNT = 64_726
V001_PROMPT_COUNTS = {
    "Reach for the fabric, grasp it, and hold it for inspection.": 126,
    "Leave the fabric on the rack, release it, and return to the ready position.": 62,
    "Remove the fabric from the rack and drop it into the basket.": 64,
}


def _require_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:  # pragma: no cover - depends on the selected runtime.
        raise RuntimeError(
            "LeRobot >= 0.4 is required. Activate the dedicated SmolVLA environment "
            "and install `pip install -e \".[smolvla]\" h5py`."
        ) from exc
    return LeRobotDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_vector(value: Any, *, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != width:
        raise ValueError(f"{name} has width {array.size}, expected {width}")
    return array


def _nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> np.ndarray:
    if source_ns.ndim != 1 or target_ns.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if source_ns.size == 0:
        raise ValueError("source timestamps are empty")
    if np.any(np.diff(source_ns) < 0):
        raise ValueError("source timestamps are not sorted")
    insertion = np.searchsorted(source_ns, target_ns)
    right = np.minimum(insertion, source_ns.size - 1)
    left = np.maximum(insertion - 1, 0)
    choose_left = np.abs(source_ns[left] - target_ns) <= np.abs(source_ns[right] - target_ns)
    return np.where(choose_left, left, right)


def _read_video(path: Path) -> list[np.ndarray]:
    """Decode once, resize to the fixed policy input source resolution, and RGB-convert."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to decode video: {path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if tuple(frame.shape[:2]) != (180, 320):
                frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            frames.append(frame[..., ::-1].copy())  # OpenCV BGR -> RGB
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"video contains no readable frames: {path}")
    return frames


def _load_episode(episode_dir: Path) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[np.ndarray]]:
    trajectory_path = episode_dir / "trajectory.h5"
    with h5py.File(trajectory_path, "r") as handle:
        arrays = {
            "timestamps": np.asarray(handle["observation/timestamp/monotonic_ns"], dtype=np.int64),
            "joint_position": np.asarray(handle["observation/robot_state/joint_positions"], dtype=np.float32),
            "gripper_position": np.asarray(handle["observation/robot_state/gripper_position"], dtype=np.float32),
            "joint_velocity": np.asarray(handle["action/joint_velocity"], dtype=np.float32),
            "commanded_gripper_position": np.asarray(handle["action/gripper_position"], dtype=np.float32),
        }

    length_fields = (
        "timestamps",
        "joint_position",
        "gripper_position",
        "joint_velocity",
        "commanded_gripper_position",
    )
    lengths = {field: len(arrays[field]) for field in length_fields}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{episode_dir.name}: inconsistent trajectory lengths: {lengths}")
    if arrays["joint_position"].ndim != 2 or arrays["joint_position"].shape[1] != 7:
        raise ValueError(f"{episode_dir.name}: joint_position must have shape (N, 7)")
    if arrays["joint_velocity"].ndim != 2 or arrays["joint_velocity"].shape[1] != 7:
        raise ValueError(f"{episode_dir.name}: joint_velocity must have shape (N, 7)")

    recordings = episode_dir / "recordings"
    exterior = _read_video(recordings / "MP4" / "exterior_image_1_left.mp4")
    wrist = _read_video(recordings / "MP4" / "wrist_image_left.mp4")
    with np.load(recordings / "timestamps" / "exterior_image_1_left.npz") as payload:
        exterior_ns = np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)
    with np.load(recordings / "timestamps" / "wrist_image_left.npz") as payload:
        wrist_ns = np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)

    if len(exterior) != len(exterior_ns):
        raise ValueError(
            f"{episode_dir.name}: exterior frame/timestamp mismatch ({len(exterior)} != {len(exterior_ns)})"
        )
    if len(wrist) != len(wrist_ns):
        raise ValueError(f"{episode_dir.name}: wrist frame/timestamp mismatch ({len(wrist)} != {len(wrist_ns)})")

    arrays["exterior_indices"] = _nearest_indices(exterior_ns, arrays["timestamps"])
    arrays["wrist_indices"] = _nearest_indices(wrist_ns, arrays["timestamps"])
    return arrays, exterior, wrist


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(table.column_names)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("manifest has no rows")

    bad_rows = [
        str(row.get("segment_id", "<unknown>"))
        for row in rows
        if row["split"] != "train" or row["dataset_decision"] != "keep"
    ]
    if bad_rows:
        preview = ", ".join(bad_rows[:5])
        raise ValueError(
            "This converter intentionally accepts only the curated training export. "
            f"Found non-train/non-kept segments, including: {preview}"
        )

    segment_ids = [str(row["segment_id"]) for row in rows]
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("manifest has duplicate segment_id values")

    for row in rows:
        if not str(row["prompt"]).strip():
            raise ValueError(f"{row['segment_id']}: empty task prompt")
        if int(row["end_ns"]) < int(row["start_ns"]):
            raise ValueError(f"{row['segment_id']}: end_ns precedes start_ns")
    return rows


def _v001_manifest_contract(
    path: Path, rows: list[dict[str, Any]], *, enforce: bool
) -> dict[str, Any]:
    """Prove this is the precise train-only export used by the prior run."""
    manifest_sha256 = _sha256(path)
    segment_counts = Counter(str(row["segment_type"]) for row in rows)
    action_branch_counts = Counter(str(row["action_branch"]) for row in rows)
    prompt_counts = Counter(str(row["prompt"]) for row in rows)
    source_episode_count = len({str(row["source_episode_id"]) for row in rows})

    actual = {
        "manifest_sha256": manifest_sha256,
        "segment_count": len(rows),
        "source_episode_count": source_episode_count,
        "segment_counts": dict(sorted(segment_counts.items())),
        "action_branch_counts": dict(sorted(action_branch_counts.items())),
        "prompt_counts": dict(sorted(prompt_counts.items())),
    }
    expected = {
        "manifest_sha256": V001_MANIFEST_SHA256,
        "segment_count": V001_SEGMENT_COUNT,
        "source_episode_count": V001_SOURCE_EPISODE_COUNT,
        "segment_counts": V001_SEGMENT_COUNTS,
        "action_branch_counts": V001_ACTION_BRANCH_COUNTS,
        "prompt_counts": V001_PROMPT_COUNTS,
    }
    matches_v001 = actual == expected
    if enforce and not matches_v001:
        raise ValueError(
            "manifest does not match the immutable Fabric-DROID v001 Pi0.5 training export. "
            f"Expected {expected}, got {actual}. Use --allow-unpinned-manifest only for an intentional new export."
        )
    actual["matches_v001"] = matches_v001
    actual["v001_contract_enforced"] = enforce
    return actual


def _features() -> dict[str, dict[str, Any]]:
    image_feature = {
        "dtype": "video",
        "shape": (180, 320, 3),
        "names": ["height", "width", "channel"],
    }
    return {
        EXTERIOR_KEY: image_feature,
        WRIST_KEY: dict(image_feature),
        STATE_KEY: {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "joint_1_position",
                "joint_2_position",
                "joint_3_position",
                "joint_4_position",
                "joint_5_position",
                "joint_6_position",
                "joint_7_position",
                "gripper_position",
            ],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (8,),
            "names": [
                "joint_1_velocity",
                "joint_2_velocity",
                "joint_3_velocity",
                "joint_4_velocity",
                "joint_5_velocity",
                "joint_6_velocity",
                "joint_7_velocity",
                "commanded_gripper_position",
            ],
        },
    }


def _to_report_value(value: Any) -> Any:
    """Make PyArrow/numpy scalar values safe for json.dumps."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_dataset(dataset_root: Path, repo_id: str) -> dict[str, Any]:
    """Load one output with LeRobot and prove it has the SmolVLA input contract."""
    LeRobotDataset = _require_lerobot()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing LeRobot v3 metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(f"dataset codebase_version is {version!r}, expected a LeRobot v3 dataset")

    features = info.get("features", {})
    expected_shapes = {
        EXTERIOR_KEY: (180, 320, 3),
        WRIST_KEY: (180, 320, 3),
        STATE_KEY: (8,),
        ACTION_KEY: (8,),
    }
    for key, shape in expected_shapes.items():
        if key not in features:
            raise ValueError(f"missing required feature: {key}")
        actual_shape = tuple(features[key].get("shape", ()))
        if actual_shape != shape:
            raise ValueError(f"{key} has shape {actual_shape}, expected {shape}")

    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav")
    if len(dataset) == 0 or dataset.meta.total_episodes == 0:
        raise ValueError("LeRobot dataset is empty")
    sample = dataset[0]
    state = np.asarray(sample[STATE_KEY])
    action = np.asarray(sample[ACTION_KEY])
    if state.shape != (8,) or action.shape != (8,):
        raise ValueError(f"unexpected sample shapes: state={state.shape}, action={action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("sample contains non-finite state/action values")
    if not isinstance(sample.get("task"), str) or not sample["task"].strip():
        raise ValueError("sample has no usable task string")

    tasks = [str(task) for task in dataset.meta.tasks.index.tolist()]
    return {
        "pass": True,
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "codebase_version": version,
        "episode_count": int(dataset.meta.total_episodes),
        "frame_count": int(dataset.meta.total_frames),
        "task_count": len(tasks),
        "tasks": tasks,
        "sample": {
            "task": sample["task"],
            "state_shape": list(state.shape),
            "action_shape": list(action.shape),
            "exterior_shape": list(np.asarray(sample[EXTERIOR_KEY]).shape),
            "wrist_shape": list(np.asarray(sample[WRIST_KEY]).shape),
        },
    }


def convert(
    manifest: Path,
    raw_root: Path,
    output_root: Path,
    repo_id: str,
    *,
    minimum_frames: int,
    parallel_encoding: bool,
    video_codec: str,
    allow_unpinned_manifest: bool,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"refusing to write to existing output {output_root}; source data is untouched, "
            "choose a new directory or validate the existing one with --validate-only"
        )
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"raw data root not found: {raw_root}")

    rows = _read_manifest(manifest)
    manifest_contract = _v001_manifest_contract(
        manifest, rows, enforce=not allow_unpinned_manifest
    )
    by_source_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source_episode[str(row["source_episode_id"])].append(row)

    # Confirm every row points at a finalized, non-hidden input before creating output.
    for episode_id in by_source_episode:
        episode_dir = raw_root / episode_id
        if episode_dir.name.startswith("."):
            raise ValueError(f"manifest unexpectedly references unfinished episode: {episode_id}")
        if not (episode_dir / "trajectory.h5").is_file():
            raise FileNotFoundError(f"{episode_id}: trajectory.h5 is missing")

    # Build beside the requested root and atomically publish only after metadata
    # and a LeRobot readback both pass. An interrupted conversion therefore never
    # looks like a reusable dataset to the training launcher.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.building-{uuid4().hex}"

    LeRobotDataset = _require_lerobot()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=staging_root,
        robot_type="franka",
        fps=FPS,
        features=_features(),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        vcodec=video_codec,
    )

    report_rows: list[dict[str, Any]] = []
    segment_counts: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    total_frames = 0
    try:
        for episode_id in sorted(by_source_episode):
            episode_dir = raw_root / episode_id
            arrays, exterior_frames, wrist_frames = _load_episode(episode_dir)
            timestamps = arrays["timestamps"]
            for row in sorted(by_source_episode[episode_id], key=lambda item: int(item["start_ns"])):
                indices = np.flatnonzero(
                    (timestamps >= int(row["start_ns"])) & (timestamps <= int(row["end_ns"]))
                )
                if indices.size < minimum_frames:
                    raise ValueError(
                        f"{row['segment_id']}: only {indices.size} policy frames, "
                        f"need at least {minimum_frames}"
                    )

                output_episode_index = int(dataset.meta.total_episodes)
                prompt = str(row["prompt"])
                for index in indices:
                    state = np.concatenate(
                        [
                            _as_vector(arrays["joint_position"][index], width=7, name="joint_position"),
                            _as_vector(arrays["gripper_position"][index], width=1, name="gripper_position"),
                        ]
                    ).astype(np.float32)
                    action = np.concatenate(
                        [
                            _as_vector(arrays["joint_velocity"][index], width=7, name="joint_velocity"),
                            _as_vector(
                                arrays["commanded_gripper_position"][index],
                                width=1,
                                name="commanded_gripper_position",
                            ),
                        ]
                    ).astype(np.float32)
                    dataset.add_frame(
                        {
                            EXTERIOR_KEY: exterior_frames[int(arrays["exterior_indices"][index])],
                            WRIST_KEY: wrist_frames[int(arrays["wrist_indices"][index])],
                            STATE_KEY: state,
                            ACTION_KEY: action,
                            "task": prompt,
                        }
                    )
                dataset.save_episode(parallel_encoding=parallel_encoding)

                frame_count = int(indices.size)
                total_frames += frame_count
                segment_type = str(row["segment_type"])
                action_branch = str(row["action_branch"])
                segment_counts[segment_type] += 1
                branch_counts[action_branch] += 1
                report_rows.append(
                    {
                        "output_episode_index": output_episode_index,
                        "segment_id": str(row["segment_id"]),
                        "source_episode_id": episode_id,
                        "segment_type": segment_type,
                        "action_branch": action_branch,
                        "prompt": prompt,
                        "frame_count": frame_count,
                        "start_ns": _to_report_value(row["start_ns"]),
                        "end_ns": _to_report_value(row["end_ns"]),
                    }
                )
    finally:
        # A failed conversion must still leave valid file footers for diagnosis.
        dataset.finalize()

    if not allow_unpinned_manifest and total_frames != V001_FRAME_COUNT:
        raise ValueError(
            "raw data does not reproduce the v001 Pi0.5 training frame count: "
            f"got {total_frames}, expected {V001_FRAME_COUNT}"
        )

    report = {
        "schema": "fabric-droid-smolvla-v3-conversion-v1",
        "repo_id": repo_id,
        "output_root": str(output_root),
        "manifest": str(manifest),
        "manifest_contract": manifest_contract,
        "raw_root": str(raw_root),
        "fps": FPS,
        "episode_count": len(report_rows),
        "frame_count": total_frames,
        "segment_counts": dict(sorted(segment_counts.items())),
        "action_branch_counts": dict(sorted(branch_counts.items())),
        "features": _features(),
        "camera_mapping_for_smolvla_base": {
            EXTERIOR_KEY: "observation.images.camera1",
            WRIST_KEY: "observation.images.camera2",
        },
        "episodes": report_rows,
    }
    report_path = staging_root / "smolvla_conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    validation = validate_dataset(staging_root, repo_id)
    validation["dataset_root"] = str(output_root)
    report["validation"] = validation
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(staging_root, output_root)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Curator manifest.parquet from the Pi0.5 export")
    parser.add_argument("--raw-root", type=Path, help="Root containing finalized Fabric-DROID episodes")
    parser.add_argument("--output-root", type=Path, required=True, help="New, empty LeRobot v3 dataset directory")
    parser.add_argument("--repo-id", default="local/fabric_droid_v001_smolvla_v3")
    parser.add_argument("--minimum-frames", type=int, default=16)
    parser.add_argument(
        "--parallel-encoding",
        action="store_true",
        help="Encode the exterior and wrist videos of each segment in parallel.",
    )
    parser.add_argument("--video-codec", choices=("h264", "hevc", "libsvtav1"), default="h264")
    parser.add_argument(
        "--allow-unpinned-manifest",
        action="store_true",
        help="Allow a different manifest. The default locks the exact v001 Pi0.5 training export.",
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing V3 output without writing.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.minimum_frames < 1:
        raise ValueError("--minimum-frames must be positive")

    if args.validate_only:
        report = validate_dataset(args.output_root, args.repo_id)
    else:
        if args.manifest is None or args.raw_root is None:
            raise ValueError("--manifest and --raw-root are required unless --validate-only is used")
        report = convert(
            args.manifest,
            args.raw_root,
            args.output_root,
            args.repo_id,
            minimum_frames=args.minimum_frames,
            parallel_encoding=args.parallel_encoding,
            video_codec=args.video_codec,
            allow_unpinned_manifest=args.allow_unpinned_manifest,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
