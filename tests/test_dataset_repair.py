from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from fabric_droid.repair import _ffprobe, repair_episode


def _write_video(path: Path, *, frames: int = 30, fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    writer.release()


def test_repair_episode_retimes_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source" / "episode_001"
    destination = tmp_path / "repaired" / "episode_001"
    mp4 = source / "recordings" / "MP4"
    timestamps = source / "recordings" / "timestamps"
    tactile = source / "tactile"
    timestamps.mkdir(parents=True)
    tactile.mkdir(parents=True)
    exterior = mp4 / "exterior_image_1_left.mp4"
    wrist = mp4 / "wrist_image_left.mp4"
    _write_video(exterior)
    shutil.copyfile(exterior, wrist)
    os.link(exterior, mp4 / "exterior_image_2_left.mp4")
    values = 1_000_000_000 + np.arange(30, dtype=np.int64) * 40_000_000
    for name in ("exterior_image_1_left", "wrist_image_left"):
        np.savez(
            timestamps / f"{name}.npz",
            timestamp_monotonic_ns=values,
        )
    (source / "trajectory.h5").write_bytes(b"test-trajectory")
    (source / "COMPLETE.json").write_text('{"complete":true}', encoding="utf-8")
    (source / "metadata_episode_001.json").write_text(
        '{"episode_id":"episode_001"}',
        encoding="utf-8",
    )
    (tactile / "events.json").write_text(
        json.dumps(
            {
                "events": [
                    {"name": "probe_complete"},
                    {"name": "release_time"},
                ]
            }
        ),
        encoding="utf-8",
    )
    source_video_stat = (exterior.stat().st_size, exterior.stat().st_mtime_ns)
    source_metadata_inode = (source / "metadata_episode_001.json").stat().st_ino

    label = repair_episode(source, destination)

    assert (exterior.stat().st_size, exterior.stat().st_mtime_ns) == source_video_stat
    repaired = _ffprobe(
        destination / "recordings/MP4/exterior_image_1_left.mp4"
    )
    assert repaired["frame_count"] == 30
    assert repaired["fps"] == pytest.approx(25.0, abs=0.15)
    assert repaired["duration_sec"] == pytest.approx(1.2, abs=0.08)
    assert label["quality_grade"] == "B_REPAIRED"
    assert label["recommendation"] == "TRAIN_FULL_AND_SEGMENTED"
    assert label["training_eligibility"]["full_trajectory"] is True
    assert label["training_eligibility"]["event_segmented"] is True
    assert label["checks"]["source_files_checked_after_repair"] is True
    assert (
        destination / "recordings/MP4/exterior_image_1_left.mp4"
    ).stat().st_ino == (
        destination / "recordings/MP4/exterior_image_2_left.mp4"
    ).stat().st_ino
    assert (
        destination / "metadata_episode_001.json"
    ).stat().st_ino == source_metadata_inode
