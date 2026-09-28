#!/usr/bin/env python3
"""Materialize the audited source-episode split as independent LeRobot v3 roots.

LeRobot v0.4.4's trainer accepts only one offline dataset and its metadata
statistics are global to that dataset.  This tool uses LeRobot's own
``split_dataset`` implementation to copy/reindex the selected episodes and
recompute each split's ``stats.json`` from its selected episodes.  Training on
the resulting ``train`` root therefore cannot use validation normalisation
statistics.

The input full dataset is never altered.  Output is built in a sibling staging
directory and atomically published only after both train and val roots have
been read back successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_split(path: Path) -> tuple[list[int], list[int], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = payload.get("dataset_episode_indices")
    if not isinstance(indices, dict):
        raise ValueError(f"{path}: missing dataset_episode_indices; regenerate the split with --conversion-report")
    train = indices.get("train")
    val = indices.get("val")
    if not isinstance(train, list) or not isinstance(val, list):
        raise ValueError(f"{path}: train and val episode indices must be lists")
    train = [int(value) for value in train]
    val = [int(value) for value in val]
    if not train or not val or set(train) & set(val):
        raise ValueError(f"{path}: train/val episode indices are empty or overlap")

    totals = payload.get("frame_totals", {})
    expected_frames = {split: int(totals[split]) for split in ("train", "val")} if totals else {}
    return train, val, expected_frames


def _validate_split_root(
    root: Path,
    repo_id: str,
    *,
    expected_episodes: int,
    expected_frames: int | None,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    if dataset.num_episodes != expected_episodes:
        raise ValueError(f"{root}: expected {expected_episodes} episodes, found {dataset.num_episodes}")
    if expected_frames is not None and dataset.num_frames != expected_frames:
        raise ValueError(f"{root}: expected {expected_frames} frames, found {dataset.num_frames}")
    if not (root / "meta" / "stats.json").is_file():
        raise FileNotFoundError(f"{root}: split stats.json was not written")
    if len(dataset) == 0:
        raise ValueError(f"{root}: split dataset has no samples")
    sample = dataset[0]
    required = {"observation.state", "action", "task"}
    missing = required - set(sample)
    if missing:
        raise ValueError(f"{root}: missing required sample fields {sorted(missing)}")
    return {
        "root": str(root),
        "repo_id": repo_id,
        "episodes": int(dataset.num_episodes),
        "frames": int(dataset.num_frames),
        "tasks": int(dataset.meta.total_tasks),
        "stats_sha256": _sha256(root / "meta" / "stats.json"),
    }


def materialize(
    source_root: Path,
    source_repo_id: str,
    split_file: Path,
    output_root: Path,
) -> dict[str, Any]:
    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source dataset root does not exist: {source_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"split file does not exist: {split_file}")

    train_episodes, val_episodes, expected_frames = _load_split(split_file)
    source = LeRobotDataset(source_repo_id, root=source_root, video_backend="pyav")
    valid = set(range(source.meta.total_episodes))
    requested = set(train_episodes) | set(val_episodes)
    if requested != valid:
        missing = sorted(valid - requested)
        extra = sorted(requested - valid)
        raise ValueError(
            f"split must partition all source episodes exactly once; missing={missing[:5]}, extra={extra[:5]}"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.building-{uuid4().hex}"
    try:
        split_dataset(
            source,
            splits={"train": train_episodes, "val": val_episodes},
            output_dir=staging_root,
        )

        train_repo_id = f"{source_repo_id}_train"
        val_repo_id = f"{source_repo_id}_val"
        report = {
            "schema": "fabric-droid-smolvla-materialized-source-split-v1",
            "source_root": str(source_root),
            "source_repo_id": source_repo_id,
            "split_file": str(split_file),
            "split_file_sha256": _sha256(split_file),
            "source_total_episodes": int(source.meta.total_episodes),
            "train": _validate_split_root(
                staging_root / "train",
                train_repo_id,
                expected_episodes=len(train_episodes),
                expected_frames=expected_frames.get("train"),
            ),
            "val": _validate_split_root(
                staging_root / "val",
                val_repo_id,
                expected_episodes=len(val_episodes),
                expected_frames=expected_frames.get("val"),
            ),
        }
        (staging_root / "materialization_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(staging_root, output_root)
        return report
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def validate_existing(
    output_root: Path,
    source_repo_id: str,
    split_file: Path,
) -> dict[str, Any]:
    if not output_root.is_dir():
        raise FileNotFoundError(f"materialized split root does not exist: {output_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"split file does not exist: {split_file}")
    train_episodes, val_episodes, expected_frames = _load_split(split_file)
    report = {
        "schema": "fabric-droid-smolvla-materialized-source-split-v1",
        "output_root": str(output_root),
        "split_file": str(split_file),
        "split_file_sha256": _sha256(split_file),
        "train": _validate_split_root(
            output_root / "train",
            f"{source_repo_id}_train",
            expected_episodes=len(train_episodes),
            expected_frames=expected_frames.get("train"),
        ),
        "val": _validate_split_root(
            output_root / "val",
            f"{source_repo_id}_val",
            expected_episodes=len(val_episodes),
            expected_frames=expected_frames.get("val"),
        ),
    }
    materialization_report = output_root / "materialization_report.json"
    if materialization_report.is_file():
        report["materialization_report_sha256"] = _sha256(materialization_report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = (
        validate_existing(args.output_root, args.source_repo_id, args.split_file)
        if args.validate_only
        else materialize(args.source_root, args.source_repo_id, args.split_file, args.output_root)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
