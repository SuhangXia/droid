from __future__ import annotations

import sys
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from fabric_droid.capture.recorder import FabricEpisodeRecorder, RecorderOptions, default_metadata
from fabric_droid.capture.robot_trajectory import (
    RobotTelemetryBuffer,
    write_droid_trajectory_h5,
)
from fabric_droid.conversion.lerobot import convert_session
from fabric_droid.conversion.segments import build_segment_manifest
from fabric_droid.validation.episode import validate_episode
from fabric_droid.validation.events import detect_event_candidates


@pytest.fixture(scope="module")
def synthetic_episode(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fabric_droid_episode")
    metadata = default_metadata(
        "target_tray",
        "swatch_integration",
        "train",
        session_id="integration",
        episode_id="episode_integration",
    )
    options = RecorderOptions(
        output_dir=root,
        duration_sec=1.2,
        dry_run=True,
        robot_disabled=True,
        record_only=True,
        width=96,
        height=64,
        fps=15,
        gelsight_fps=30,
        minimum_free_gib=0.01,
    )
    episode = FabricEpisodeRecorder(options, metadata).record_for_duration()
    assert not (root / ".episode_integration.inprogress").exists()
    return episode


def test_atomic_capture_validation_events_and_segments(synthetic_episode: Path) -> None:
    report = validate_episode(synthetic_episode, minimum_duration_sec=1.0)
    assert report["pass"], report["failures"]
    assert report["ati"]["clock"]["count"] >= 550
    assert 450 <= report["ati"]["clock"]["measured_hz"] <= 550
    candidates = detect_event_candidates(synthetic_episode)
    assert "manual_comparison" in candidates
    segments = build_segment_manifest(synthetic_episode)
    assert segments["valid"], segments["failures"]
    assert segments["segments"][0]["timestep_start"] == 0
    assert segments["segments"][1]["timestep_start"] == 0


def test_real_telemetry_droid_h5_validates_with_sensor_episode(
    synthetic_episode: Path,
    tmp_path: Path,
) -> None:
    episode = tmp_path / synthetic_episode.name
    shutil.copytree(synthetic_episode, episode)
    with h5py.File(episode / "trajectory.h5", "r") as handle:
        timestamps = np.asarray(
            handle["observation/timestamp/monotonic_ns"],
            dtype=np.int64,
        )
    telemetry = RobotTelemetryBuffer()
    for index, timestamp_ns in enumerate(timestamps):
        telemetry.append_tick(
            {
                "timestamp_monotonic_ns": int(timestamp_ns),
                "robot_timestamp_seconds": 10,
                "robot_timestamp_nanos": index,
                "frame_sequence": index,
                "joint_positions_rad": np.full(7, index * 0.001),
                "joint_velocities_rad_s": np.full(7, 0.01),
                "measured_position_m": [0.45, 0.0, 0.5],
                "measured_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "target_position_m": [0.45, 0.0, 0.5],
                "target_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "gripper_position": min(0.9, index / max(1, len(timestamps))),
                "commanded_gripper_position": min(
                    0.9,
                    (index + 1) / max(1, len(timestamps)),
                ),
                "deadman_pressed": index > 0,
            }
        )
    telemetry_path = episode / "robot_telemetry.npz"
    telemetry.write_npz_atomic(telemetry_path)
    write_droid_trajectory_h5(
        telemetry_path,
        episode / "trajectory.h5",
        success=True,
        failure_reason="",
        task_instruction="Inspect the fabric.",
        robot_ip="192.168.0.116",
    )
    report = validate_episode(episode, minimum_duration_sec=1.0)
    assert report["pass"], report["failures"]
    assert report["robot_length"] == len(timestamps)
    assert report["state_action_finite"]


def test_lerobot_conversion_and_openpi_batch(synthetic_episode: Path, tmp_path: Path) -> None:
    pytest.importorskip("lerobot")
    output = tmp_path / "lerobot"
    report = convert_session(
        synthetic_episode.parent,
        output,
        "local/fabric_droid_integration",
        minimum_duration_sec=1.0,
    )
    assert report["episode_count"] == 1
    assert report["frame_count"] >= 15
    openpi_src = Path("/home/suhang/projects/openpi/src")
    if not openpi_src.is_dir():
        pytest.skip("local OpenPI source is unavailable")
    sys.path.insert(0, str(openpi_src))
    try:
        from fabric_droid.conversion.pi05_smoke import run_pi05_batch_smoke

        smoke = run_pi05_batch_smoke(output, "local/fabric_droid_integration")
    finally:
        sys.path.remove(str(openpi_src))
    assert smoke["pass"], smoke["failures"]
    assert smoke["pi05"]["state_shape"] == [8]
    assert smoke["pi05"]["actions_shape"] == [16, 8]
