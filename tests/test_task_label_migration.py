from __future__ import annotations

import json
from pathlib import Path

import h5py

from fabric_droid.migration.task_labels import (
    MANIFEST_NAME,
    migrate_task_labels,
)
from fabric_droid.schemas import (
    CANONICAL_DESTINATION_TRAY,
    CANONICAL_TASK_INSTRUCTION,
)


def test_task_label_migration_updates_all_active_consumers(tmp_path: Path) -> None:
    episode = tmp_path / "episode_001"
    episode.mkdir()
    (episode / "COMPLETE.json").write_text(
        '{"complete": true}\n',
        encoding="utf-8",
    )
    metadata_path = episode / "metadata_episode_001.json"
    metadata_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_001",
                "destination_tray": "legacy_side",
                "task_instruction": "Legacy instruction.",
                "robot_motion_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    assignment_path = episode / "device_assignment.json"
    assignment_path.write_text(
        json.dumps({"destination_tray": "legacy_side"}),
        encoding="utf-8",
    )
    with h5py.File(episode / "trajectory.h5", "w") as handle:
        handle.attrs["current_task"] = "Legacy instruction."

    report = migrate_task_labels(tmp_path)
    assert report["status"] == "complete"
    assert report["episode_count"] == 1
    assert report["changed_episode_count"] == 1

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    with h5py.File(episode / "trajectory.h5", "r") as handle:
        h5_task = handle.attrs["current_task"]
    assert metadata["destination_tray"] == CANONICAL_DESTINATION_TRAY
    assert metadata["task_instruction"] == CANONICAL_TASK_INSTRUCTION
    assert metadata["robot_motion_enabled"] is True
    assert assignment["destination_tray"] == CANONICAL_DESTINATION_TRAY
    assert h5_task == CANONICAL_TASK_INSTRUCTION
    assert (tmp_path / MANIFEST_NAME).is_file()

    second = migrate_task_labels(tmp_path)
    assert second["idempotent_noop"] is True
    assert second["episodes"][0]["previous"]["metadata_destination_tray"] == ("legacy_side")
