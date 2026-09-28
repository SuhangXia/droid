"""Real Franka telemetry buffering and DROID-compatible trajectory output."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np


class RobotTrajectoryError(RuntimeError):
    """Raised when real robot telemetry cannot form a valid trajectory."""


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != length:
        raise RobotTrajectoryError(f"{name} must contain {length} values, got {array.size}")
    if not np.isfinite(array).all():
        raise RobotTrajectoryError(f"{name} contains NaN or infinity")
    return array


def _quat_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _vector(quat_xyzw, 4, "quat_xyzw")
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm < 1e-8:
        raise RobotTrajectoryError("quat_xyzw norm is near zero")
    x, y, z, w = (np.asarray((x, y, z, w)) / norm).tolist()
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray((roll, pitch, yaw), dtype=np.float64)


class RobotTelemetryBuffer:
    """Keep low-dimensional 30 Hz telemetry in memory and close it atomically."""

    def __init__(self) -> None:
        self._samples: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._samples)

    def append_tick(self, tick: dict[str, Any]) -> None:
        timestamp_ns = int(tick["timestamp_monotonic_ns"])
        if self._samples and timestamp_ns <= int(self._samples[-1]["timestamp_monotonic_ns"]):
            raise RobotTrajectoryError("robot telemetry timestamps must be strictly increasing")
        gripper_position = float(tick["gripper_position"])
        commanded_gripper_position = float(tick["commanded_gripper_position"])
        if not 0.0 <= gripper_position <= 1.0:
            raise RobotTrajectoryError("measured gripper_position must be within [0, 1]")
        if not 0.0 <= commanded_gripper_position <= 1.0:
            raise RobotTrajectoryError("commanded gripper_position must be within [0, 1]")
        sample = {
            "timestamp_monotonic_ns": timestamp_ns,
            "robot_timestamp_seconds": int(tick["robot_timestamp_seconds"]),
            "robot_timestamp_nanos": int(tick["robot_timestamp_nanos"]),
            "frame_sequence": int(tick["frame_sequence"]),
            "joint_positions": _vector(tick["joint_positions_rad"], 7, "joint_positions"),
            "joint_velocities": _vector(
                tick["joint_velocities_rad_s"],
                7,
                "joint_velocities",
            ),
            "ee_position": _vector(tick["measured_position_m"], 3, "measured_position"),
            "ee_quat": _vector(tick["measured_quat_xyzw"], 4, "measured_quat"),
            "target_position": _vector(tick["target_position_m"], 3, "target_position"),
            "target_quat": _vector(tick["target_quat_xyzw"], 4, "target_quat"),
            "gripper_position": gripper_position,
            "commanded_gripper_position": commanded_gripper_position,
            "movement_enabled": bool(tick["deadman_pressed"]),
            "skip_action": not bool(tick["deadman_pressed"]),
        }
        self._samples.append(sample)

    def write_npz_atomic(self, path: Path) -> None:
        if len(self._samples) < 2:
            raise RobotTrajectoryError(
                f"real robot telemetry contains too few samples: {len(self._samples)}"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")

        def values(name: str, dtype: Any | None = None) -> np.ndarray:
            return np.asarray([sample[name] for sample in self._samples], dtype=dtype)

        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema_version=np.asarray([1], dtype=np.int64),
                timestamp_monotonic_ns=values("timestamp_monotonic_ns", np.int64),
                robot_timestamp_seconds=values("robot_timestamp_seconds", np.int64),
                robot_timestamp_nanos=values("robot_timestamp_nanos", np.int64),
                frame_sequence=values("frame_sequence", np.int64),
                joint_positions=values("joint_positions", np.float64),
                joint_velocities=values("joint_velocities", np.float64),
                ee_position=values("ee_position", np.float64),
                ee_quat=values("ee_quat", np.float64),
                target_position=values("target_position", np.float64),
                target_quat=values("target_quat", np.float64),
                gripper_position=values("gripper_position", np.float64),
                commanded_gripper_position=values(
                    "commanded_gripper_position",
                    np.float64,
                ),
                movement_enabled=values("movement_enabled", np.bool_),
                skip_action=values("skip_action", np.bool_),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def write_droid_trajectory_h5(
    telemetry_path: Path,
    output_path: Path,
    *,
    success: bool,
    failure_reason: str,
    task_instruction: str,
    robot_ip: str,
) -> dict[str, Any]:
    """Atomically convert real telemetry into the raw DROID hierarchy.

    Joint velocity actions are the Franka controller's measured/achieved joint
    velocities for each Cartesian impedance control tick. This matches the
    7D joint-velocity action convention consumed by OpenPI's pi05-DROID
    converter without inventing IK commands that Polymetis did not issue.
    """

    try:
        import h5py
    except ImportError as exc:
        raise RobotTrajectoryError("h5py is required for trajectory.h5") from exc

    telemetry_path = Path(telemetry_path)
    if not telemetry_path.is_file():
        raise RobotTrajectoryError(f"missing real robot telemetry: {telemetry_path}")
    with np.load(telemetry_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}

    required_shapes = {
        "joint_positions": (7,),
        "joint_velocities": (7,),
        "ee_position": (3,),
        "ee_quat": (4,),
        "target_position": (3,),
        "target_quat": (4,),
    }
    timestamps = np.asarray(arrays["timestamp_monotonic_ns"], dtype=np.int64).reshape(-1)
    count = int(timestamps.size)
    if count < 2:
        raise RobotTrajectoryError(f"trajectory needs at least 2 robot samples, got {count}")
    if np.any(np.diff(timestamps) <= 0):
        raise RobotTrajectoryError("robot telemetry timestamps are not strictly monotonic")
    for name, tail_shape in required_shapes.items():
        if arrays[name].shape != (count, *tail_shape):
            raise RobotTrajectoryError(
                f"{name} shape is {arrays[name].shape}, expected {(count, *tail_shape)}"
            )
        if not np.isfinite(arrays[name]).all():
            raise RobotTrajectoryError(f"{name} contains NaN or infinity")
    for name in ("gripper_position", "commanded_gripper_position"):
        arrays[name] = np.asarray(arrays[name], dtype=np.float64).reshape(-1)
        if arrays[name].shape != (count,) or not np.isfinite(arrays[name]).all():
            raise RobotTrajectoryError(f"{name} must be finite with shape ({count},)")
        if np.any((arrays[name] < 0.0) | (arrays[name] > 1.0)):
            raise RobotTrajectoryError(f"{name} must be within [0, 1]")

    measured_cartesian = np.concatenate(
        [
            np.asarray(arrays["ee_position"], dtype=np.float64),
            np.stack([_quat_to_euler_xyz(value) for value in arrays["ee_quat"]]),
        ],
        axis=1,
    )
    target_cartesian = np.concatenate(
        [
            np.asarray(arrays["target_position"], dtype=np.float64),
            np.stack([_quat_to_euler_xyz(value) for value in arrays["target_quat"]]),
        ],
        axis=1,
    )
    joint_positions = np.asarray(arrays["joint_positions"], dtype=np.float64)
    joint_velocities = np.asarray(arrays["joint_velocities"], dtype=np.float64)
    sample_period = np.diff(timestamps, append=timestamps[-1] + int(np.median(np.diff(timestamps)))) / 1e9
    commanded_joint_positions = joint_positions + joint_velocities * sample_period[:, None]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            {
                "success": bool(success),
                "failure": not bool(success),
                "failure_reason": str(failure_reason),
                "current_task": str(task_instruction),
                "robot_serial_number": f"franka@{robot_ip}",
                "version_number": "fabric-droid-ui-1",
                "action_space": "joint_velocity",
                "gripper_action_space": "position",
                "time_axis": "host_monotonic_ns",
                "control_hz_measured": float(
                    (count - 1) * 1e9 / (timestamps[-1] - timestamps[0])
                ),
                "joint_velocity_action_source": "measured_franka_joint_velocity",
            }
        )
        observation = handle.create_group("observation")
        robot_state = observation.create_group("robot_state")
        robot_state.create_dataset("joint_positions", data=joint_positions)
        robot_state.create_dataset("joint_velocities", data=joint_velocities)
        robot_state.create_dataset("cartesian_position", data=measured_cartesian)
        robot_state.create_dataset("gripper_position", data=arrays["gripper_position"])

        timestamp = observation.create_group("timestamp")
        timestamp.create_dataset("monotonic_ns", data=timestamps)
        timestamp.create_dataset("host_monotonic_ns", data=timestamps)
        timestamp.create_dataset(
            "robot_timestamp_seconds",
            data=np.asarray(arrays["robot_timestamp_seconds"], dtype=np.int64),
        )
        timestamp.create_dataset(
            "robot_timestamp_nanos",
            data=np.asarray(arrays["robot_timestamp_nanos"], dtype=np.int64),
        )
        timestamp.create_dataset(
            "skip_action",
            data=np.asarray(arrays["skip_action"], dtype=np.bool_),
        )

        controller_info = observation.create_group("controller_info")
        controller_info.create_dataset(
            "movement_enabled",
            data=np.asarray(arrays["movement_enabled"], dtype=np.bool_),
        )
        controller_info.create_dataset(
            "controller_on",
            data=np.ones(count, dtype=np.bool_),
        )
        controller_info.create_dataset("success", data=np.zeros(count, dtype=np.bool_))
        controller_info.create_dataset("failure", data=np.zeros(count, dtype=np.bool_))

        camera_type = observation.create_group("camera_type")
        camera_type.create_dataset("wrist_image_left", data=np.zeros(count, dtype=np.int8))
        camera_type.create_dataset(
            "exterior_image_1_left",
            data=np.ones(count, dtype=np.int8),
        )
        camera_type.create_dataset(
            "exterior_image_2_left",
            data=np.ones(count, dtype=np.int8),
        )

        action = handle.create_group("action")
        action.create_dataset("joint_velocity", data=joint_velocities)
        action.create_dataset("joint_position", data=commanded_joint_positions)
        action.create_dataset("cartesian_position", data=target_cartesian)
        action.create_dataset(
            "gripper_position",
            data=arrays["commanded_gripper_position"],
        )
    os.replace(temporary, output_path)
    return {
        "sample_count": count,
        "duration_sec": float((timestamps[-1] - timestamps[0]) / 1e9),
        "measured_hz": float((count - 1) * 1e9 / (timestamps[-1] - timestamps[0])),
        "action_space": "joint_velocity_7_plus_absolute_gripper_position_1",
        "gripper_convention": "0=open, 1=closed",
    }
