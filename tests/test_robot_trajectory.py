from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from droid.trajectory_utils.trajectory_reader import TrajectoryReader
from fabric_droid.capture.robot_trajectory import (
    RobotTelemetryBuffer,
    RobotTrajectoryError,
    write_droid_trajectory_h5,
)


def tick(index: int, *, timestamp_ns: int | None = None) -> dict[str, object]:
    timestamp_ns = timestamp_ns if timestamp_ns is not None else 1_000_000_000 + index * 40_000_000
    return {
        "timestamp_monotonic_ns": timestamp_ns,
        "robot_timestamp_seconds": 10,
        "robot_timestamp_nanos": index * 40_000_000,
        "frame_sequence": index,
        "joint_positions_rad": np.arange(7, dtype=np.float64) * 0.1 + index * 0.01,
        "joint_velocities_rad_s": np.full(7, 0.25 + index * 0.01),
        "measured_position_m": [0.45 + index * 0.001, 0.0, 0.5],
        "measured_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "target_position_m": [0.46 + index * 0.001, 0.0, 0.5],
        "target_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_position": 0.2 + index * 0.1,
        "commanded_gripper_position": 0.3 + index * 0.1,
        "deadman_pressed": bool(index % 2),
    }


def test_real_telemetry_writes_atomic_droid_trajectory(tmp_path: Path) -> None:
    buffer = RobotTelemetryBuffer()
    for index in range(3):
        buffer.append_tick(tick(index))
    telemetry_path = tmp_path / "robot_telemetry.npz"
    buffer.write_npz_atomic(telemetry_path)
    assert telemetry_path.is_file()
    assert not (tmp_path / ".robot_telemetry.npz.tmp").exists()

    trajectory_path = tmp_path / "trajectory.h5"
    report = write_droid_trajectory_h5(
        telemetry_path,
        trajectory_path,
        success=True,
        failure_reason="",
        task_instruction="Inspect the fabric.",
        robot_ip="192.168.0.116",
    )
    assert report["sample_count"] == 3
    assert report["action_space"].startswith("joint_velocity_7")
    assert not (tmp_path / ".trajectory.h5.tmp").exists()

    with h5py.File(trajectory_path, "r") as handle:
        assert bool(handle.attrs["success"])
        assert handle.attrs["current_task"] == "Inspect the fabric."
        assert handle["observation/robot_state/joint_positions"].shape == (3, 7)
        assert handle["observation/robot_state/gripper_position"][:].tolist() == pytest.approx(
            [0.2, 0.3, 0.4]
        )
        assert handle["action/joint_velocity"].shape == (3, 7)
        assert handle["action/gripper_position"][:].tolist() == pytest.approx(
            [0.3, 0.4, 0.5]
        )
        assert set(handle["observation/camera_type"]) == {
            "wrist_image_left",
            "exterior_image_1_left",
            "exterior_image_2_left",
        }
        assert handle["observation/timestamp/skip_action"][:].tolist() == [
            True,
            False,
            True,
        ]

    reader = TrajectoryReader(str(trajectory_path), read_images=False)
    try:
        assert reader.length() == 3
        first = reader.read_timestep()
    finally:
        reader.close()
    assert first["observation"]["robot_state"]["joint_positions"].shape == (7,)
    assert first["action"]["joint_velocity"].shape == (7,)


def test_telemetry_rejects_nonmonotonic_time_and_invalid_gripper() -> None:
    buffer = RobotTelemetryBuffer()
    buffer.append_tick(tick(0))
    with pytest.raises(RobotTrajectoryError, match="strictly increasing"):
        buffer.append_tick(tick(1, timestamp_ns=1_000_000_000))

    invalid = tick(2)
    invalid["gripper_position"] = 1.1
    with pytest.raises(RobotTrajectoryError, match="within"):
        RobotTelemetryBuffer().append_tick(invalid)


def test_telemetry_refuses_too_few_samples(tmp_path: Path) -> None:
    buffer = RobotTelemetryBuffer()
    buffer.append_tick(tick(0))
    with pytest.raises(RobotTrajectoryError, match="too few"):
        buffer.write_npz_atomic(tmp_path / "robot_telemetry.npz")
