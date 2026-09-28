"""Migrate legacy color/direction task labels to the canonical task."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from fabric_droid.io_utils import atomic_write_json
from fabric_droid.schemas import (
    CANONICAL_DESTINATION_TRAY,
    CANONICAL_TASK_INSTRUCTION,
)

MIGRATION_SCHEMA = "fabric-droid-task-label-migration-v1"
MANIFEST_NAME = "TASK_LABEL_MIGRATION.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        return value.item()
    return value


def _scan_dataset(dataset_root: Path) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to migrate trajectory.h5 labels") from exc

    episode_dirs = sorted(
        path for path in dataset_root.glob("episode_*") if path.is_dir() and (path / "COMPLETE.json").is_file()
    )
    if not episode_dirs:
        raise ValueError(f"no complete episode_* directories found in {dataset_root}")

    records: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        metadata_paths = sorted(episode_dir.glob("metadata_*.json"))
        if len(metadata_paths) != 1:
            raise ValueError(f"{episode_dir.name}: expected one metadata_*.json, " f"found {len(metadata_paths)}")
        metadata_path = metadata_paths[0]
        assignment_path = episode_dir / "device_assignment.json"
        trajectory_path = episode_dir / "trajectory.h5"
        if not assignment_path.is_file():
            raise FileNotFoundError(f"{episode_dir.name}: missing device_assignment.json")
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"{episode_dir.name}: missing trajectory.h5")

        metadata = _read_json(metadata_path)
        assignment = _read_json(assignment_path)
        if str(metadata.get("episode_id")) != episode_dir.name:
            raise ValueError(f"{episode_dir.name}: metadata episode_id is " f"{metadata.get('episode_id')!r}")
        with h5py.File(trajectory_path, "r") as handle:
            if "current_task" not in handle.attrs:
                raise ValueError(f"{episode_dir.name}: missing H5 current_task attribute")
            h5_task = _json_value(handle.attrs["current_task"])

        records.append(
            {
                "episode_id": episode_dir.name,
                "metadata_path": metadata_path,
                "assignment_path": assignment_path,
                "trajectory_path": trajectory_path,
                "metadata": metadata,
                "assignment": assignment,
                "previous": {
                    "metadata_destination_tray": metadata.get("destination_tray"),
                    "metadata_task_instruction": metadata.get("task_instruction"),
                    "metadata_robot_motion_enabled": metadata.get("robot_motion_enabled"),
                    "assignment_destination_tray": assignment.get("destination_tray"),
                    "h5_current_task": h5_task,
                },
            }
        )
    return records


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    previous = record["previous"]
    changed_fields = [
        name
        for name, value, expected in (
            (
                "metadata.destination_tray",
                previous["metadata_destination_tray"],
                CANONICAL_DESTINATION_TRAY,
            ),
            (
                "metadata.task_instruction",
                previous["metadata_task_instruction"],
                CANONICAL_TASK_INSTRUCTION,
            ),
            (
                "device_assignment.destination_tray",
                previous["assignment_destination_tray"],
                CANONICAL_DESTINATION_TRAY,
            ),
            (
                "trajectory.h5/current_task",
                previous["h5_current_task"],
                CANONICAL_TASK_INSTRUCTION,
            ),
        )
        if value != expected
    ]
    if previous["metadata_robot_motion_enabled"] is not True:
        changed_fields.append("metadata.robot_motion_enabled")
    return {
        "episode_id": record["episode_id"],
        "previous": previous,
        "changed_fields": changed_fields,
    }


def _write_h5_task(path: Path, value: Any) -> None:
    import h5py

    with h5py.File(path, "r+") as handle:
        handle.attrs["current_task"] = value
        handle.flush()


def _verify_canonical(records: list[dict[str, Any]]) -> list[str]:
    import h5py

    failures: list[str] = []
    for record in records:
        metadata = _read_json(record["metadata_path"])
        assignment = _read_json(record["assignment_path"])
        with h5py.File(record["trajectory_path"], "r") as handle:
            h5_task = _json_value(handle.attrs.get("current_task"))
        if metadata.get("destination_tray") != CANONICAL_DESTINATION_TRAY:
            failures.append(f"{record['episode_id']}: metadata destination")
        if metadata.get("task_instruction") != CANONICAL_TASK_INSTRUCTION:
            failures.append(f"{record['episode_id']}: metadata instruction")
        if metadata.get("robot_motion_enabled") is not True:
            failures.append(f"{record['episode_id']}: robot_motion_enabled")
        if assignment.get("destination_tray") != CANONICAL_DESTINATION_TRAY:
            failures.append(f"{record['episode_id']}: device assignment")
        if h5_task != CANONICAL_TASK_INSTRUCTION:
            failures.append(f"{record['episode_id']}: H5 current_task")
    return failures


def migrate_task_labels(
    dataset_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Migrate active task fields with atomic JSON writes and rollback audit."""

    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(dataset_root)
    records = _scan_dataset(dataset_root)
    public_records = [_public_record(record) for record in records]
    changed_episode_count = sum(bool(record["changed_fields"]) for record in public_records)
    manifest: dict[str, Any] = {
        "schema": MIGRATION_SCHEMA,
        "status": "dry_run" if dry_run else "in_progress",
        "dataset_root": str(dataset_root),
        "created_wall_time_ns": time.time_ns(),
        "canonical": {
            "destination_tray": CANONICAL_DESTINATION_TRAY,
            "task_instruction": CANONICAL_TASK_INSTRUCTION,
        },
        "episode_count": len(records),
        "changed_episode_count": changed_episode_count,
        "episodes": public_records,
    }
    if dry_run:
        return manifest

    manifest_path = dataset_root / MANIFEST_NAME
    if manifest_path.is_file() and changed_episode_count == 0:
        previous_manifest = _read_json(manifest_path)
        if previous_manifest.get("schema") == MIGRATION_SCHEMA and previous_manifest.get("status") == "complete":
            return {
                **previous_manifest,
                "idempotent_noop": True,
            }
    atomic_write_json(manifest_path, manifest)
    applied: list[dict[str, Any]] = []
    try:
        for record in records:
            # Register the episode before its first write so a failure between
            # the two JSON updates and the H5 attribute update is recoverable.
            applied.append(record)
            metadata = copy.deepcopy(record["metadata"])
            assignment = copy.deepcopy(record["assignment"])
            metadata["destination_tray"] = CANONICAL_DESTINATION_TRAY
            metadata["task_instruction"] = CANONICAL_TASK_INSTRUCTION
            metadata["robot_motion_enabled"] = True
            assignment["destination_tray"] = CANONICAL_DESTINATION_TRAY
            atomic_write_json(record["metadata_path"], metadata)
            atomic_write_json(record["assignment_path"], assignment)
            _write_h5_task(
                record["trajectory_path"],
                CANONICAL_TASK_INSTRUCTION,
            )
        verification_failures = _verify_canonical(records)
        if verification_failures:
            raise RuntimeError("post-migration verification failed: " + "; ".join(verification_failures))
    except BaseException as exc:
        rollback_failures: list[str] = []
        for record in reversed(applied):
            try:
                atomic_write_json(record["metadata_path"], record["metadata"])
                atomic_write_json(record["assignment_path"], record["assignment"])
                _write_h5_task(
                    record["trajectory_path"],
                    record["previous"]["h5_current_task"],
                )
            except BaseException as rollback_exc:
                rollback_failures.append(f"{record['episode_id']}: {rollback_exc}")
        manifest["status"] = "rollback_failed" if rollback_failures else "rolled_back"
        manifest["error"] = str(exc)
        manifest["rollback_failures"] = rollback_failures
        manifest["finished_wall_time_ns"] = time.time_ns()
        atomic_write_json(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["verification"] = {
        "pass": True,
        "metadata_count": len(records),
        "device_assignment_count": len(records),
        "h5_current_task_count": len(records),
    }
    manifest["finished_wall_time_ns"] = time.time_ns()
    atomic_write_json(manifest_path, manifest)
    return manifest
