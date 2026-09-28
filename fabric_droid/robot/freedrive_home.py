"""Guarded low-impedance hand-guiding and Franka home-pose capture.

This module deliberately does not import Polymetis or PyTorch at import time.
The command-line entry point imports those dependencies only after its explicit
robot-control gates have passed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from fabric_droid.io_utils import atomic_write_json


class FreedriveError(RuntimeError):
    """Raised when a hand-guiding safety or control invariant is violated."""


class RobotNotStationaryError(FreedriveError):
    """Raised when preflight observes joint motion above the stationary limit."""


@dataclass(frozen=True)
class LowImpedanceConfig:
    """Validated low Cartesian impedance settings for hand guiding."""

    stiffness: tuple[float, float, float, float, float, float] = (
        20.0,
        20.0,
        20.0,
        2.0,
        2.0,
        2.0,
    )
    damping: tuple[float, float, float, float, float, float] = (
        7.0,
        7.0,
        7.0,
        0.8,
        0.8,
        0.8,
    )
    follow_hz: float = 15.0
    max_duration_sec: float = 300.0

    def validate(self) -> None:
        stiffness = _finite_vector(self.stiffness, length=6, name="stiffness")
        damping = _finite_vector(self.damping, length=6, name="damping")
        if np.any(stiffness < 0.0) or np.any(damping < 0.0):
            raise FreedriveError("stiffness and damping must be non-negative")
        if np.any(stiffness[:3] > 50.0) or np.any(stiffness[3:] > 5.0):
            raise FreedriveError(
                "refusing non-low Cartesian stiffness; limits are 50 N/m translation "
                "and 5 Nm/rad rotation"
            )
        if np.any(damping[:3] > 15.0) or np.any(damping[3:] > 3.0):
            raise FreedriveError(
                "refusing non-low Cartesian damping; limits are 15 translation and 3 rotation"
            )
        if not math.isfinite(self.follow_hz) or not 5.0 <= self.follow_hz <= 60.0:
            raise FreedriveError("follow_hz must be within [5, 60]")
        if not math.isfinite(self.max_duration_sec) or not 1.0 <= self.max_duration_sec <= 1800.0:
            raise FreedriveError("max_duration_sec must be within [1, 1800]")


@dataclass(frozen=True)
class HomeSnapshot:
    """A validated Franka flange pose and joint configuration."""

    position_m: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    joint_positions_rad: tuple[float, float, float, float, float, float, float]
    robot_timestamp_seconds: Optional[int] = None
    robot_timestamp_nanos: Optional[int] = None


def _finite_vector(values: Sequence[float], *, length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != length:
        raise FreedriveError(f"{name} must contain {length} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise FreedriveError(f"{name} contains NaN or infinity: {array.tolist()}")
    return array


def _normalise_quaternion(quat_xyzw: Sequence[float]) -> np.ndarray:
    quat = _finite_vector(quat_xyzw, length=4, name="quat_xyzw")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise FreedriveError("quat_xyzw norm is near zero")
    return quat / norm


def _rotation_from_quaternion(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalise_quaternion(quat_xyzw)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def preflight_robot(robot: Any, *, max_stationary_velocity_rad_s: float = 0.05) -> dict[str, Any]:
    """Read and validate robot state without starting or replacing a policy."""

    if robot.is_running_policy():
        raise FreedriveError(
            "a Polymetis policy is already running; refusing to replace or terminate an unknown controller"
        )
    state = robot.get_robot_state()
    joints = _finite_vector(state.joint_positions, length=7, name="joint_positions")
    velocities = _finite_vector(state.joint_velocities, length=7, name="joint_velocities")
    error_code = int(getattr(state, "error_code", 0))
    if error_code != 0:
        raise FreedriveError(f"robot state error_code is {error_code}, expected 0")
    if not bool(getattr(state, "prev_command_successful", True)):
        raise FreedriveError("previous robot command was not successful")
    max_velocity = float(np.max(np.abs(velocities)))
    if max_velocity > max_stationary_velocity_rad_s:
        raise RobotNotStationaryError(
            f"robot is not stationary: max joint velocity {max_velocity:.4f} rad/s exceeds "
            f"{max_stationary_velocity_rad_s:.4f}"
        )

    position, quat = robot.get_ee_pose()
    position_np = _finite_vector(_to_list(position), length=3, name="ee_position")
    quat_np = _normalise_quaternion(_to_list(quat))
    timestamp = getattr(state, "timestamp", None)
    return {
        "joint_positions_rad": joints.tolist(),
        "joint_velocities_rad_s": velocities.tolist(),
        "max_abs_joint_velocity_rad_s": max_velocity,
        "ee_position_m": position_np.tolist(),
        "ee_quat_xyzw": quat_np.tolist(),
        "error_code": error_code,
        "prev_command_successful": bool(getattr(state, "prev_command_successful", True)),
        "robot_timestamp_seconds": getattr(timestamp, "seconds", None),
        "robot_timestamp_nanos": getattr(timestamp, "nanos", None),
    }


def wait_for_stationary_preflight(
    robot: Any,
    *,
    timeout_sec: float = 3.0,
    poll_interval_sec: float = 0.1,
    max_stationary_velocity_rad_s: float = 0.05,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run preflight, briefly retrying only a moving-robot result.

    Active/unknown policies, robot errors, and unsuccessful commands are never
    retried or suppressed. This is intended for the short residual velocity
    seen immediately after a previous policy has stopped.
    """

    if not math.isfinite(timeout_sec) or not 0.0 <= timeout_sec <= 5.0:
        raise FreedriveError("stationary preflight timeout_sec must be within [0, 5]")
    if not math.isfinite(poll_interval_sec) or poll_interval_sec <= 0.0:
        raise FreedriveError("stationary preflight poll_interval_sec must be positive")

    deadline = monotonic() + timeout_sec
    while True:
        try:
            return preflight_robot(
                robot,
                max_stationary_velocity_rad_s=max_stationary_velocity_rad_s,
            )
        except RobotNotStationaryError:
            remaining = deadline - monotonic()
            if remaining <= 0.0:
                raise
            sleep(min(poll_interval_sec, remaining))


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy()).reshape(-1).astype(np.float64).tolist()
    if hasattr(value, "tolist"):
        result = value.tolist()
        return np.asarray(result).reshape(-1).astype(np.float64).tolist()
    return np.asarray(value).reshape(-1).astype(np.float64).tolist()


def capture_home_snapshot(robot: Any) -> HomeSnapshot:
    """Capture one coherent-enough read-only home snapshot after hand guiding."""

    position, quat = robot.get_ee_pose()
    state = robot.get_robot_state()
    position_np = _finite_vector(_to_list(position), length=3, name="ee_position")
    quat_np = _normalise_quaternion(_to_list(quat))
    joints = _finite_vector(state.joint_positions, length=7, name="joint_positions")
    timestamp = getattr(state, "timestamp", None)
    return HomeSnapshot(
        position_m=tuple(float(value) for value in position_np),
        quat_xyzw=tuple(float(value) for value in quat_np),
        joint_positions_rad=tuple(float(value) for value in joints),
        robot_timestamp_seconds=getattr(timestamp, "seconds", None),
        robot_timestamp_nanos=getattr(timestamp, "nanos", None),
    )


def home_snapshot_payload(
    snapshot: HomeSnapshot,
    *,
    robot_ip: str,
    source: str,
    config: LowImpedanceConfig,
) -> dict[str, Any]:
    """Return a template-compatible, extended home-pose JSON payload."""

    position = _finite_vector(snapshot.position_m, length=3, name="position_m")
    quat = _normalise_quaternion(snapshot.quat_xyzw)
    joints = _finite_vector(snapshot.joint_positions_rad, length=7, name="joint_positions_rad")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_quaternion(quat)
    transform[:3, 3] = position
    return {
        "schema_version": 1,
        "name": "franka_home_pose",
        "frame": "franka_base",
        "body": "flange",
        "source": source,
        "robot_ip": robot_ip,
        "timestamp": time.time(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "position": position.tolist(),
        "quat_xyzw": quat.tolist(),
        "T_base_flange": transform.tolist(),
        "joint_positions_rad": joints.tolist(),
        "robot_timestamp": {
            "seconds": snapshot.robot_timestamp_seconds,
            "nanos": snapshot.robot_timestamp_nanos,
        },
        "capture_mode": {
            "name": "low_cartesian_impedance_hand_guiding",
            "stiffness": list(config.stiffness),
            "damping": list(config.damping),
            "follow_hz": config.follow_hz,
        },
    }


def save_home_snapshot(
    path: Path,
    snapshot: HomeSnapshot,
    *,
    robot_ip: str,
    source: str,
    config: LowImpedanceConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically save a home snapshot, refusing accidental replacement."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"home pose already exists; pass --overwrite-home to replace it: {path}")
    payload = home_snapshot_payload(snapshot, robot_ip=robot_ip, source=source, config=config)
    atomic_write_json(path, payload)
    return payload


def run_low_impedance_freedrive(
    robot: Any,
    *,
    config: LowImpedanceConfig,
    should_save: Callable[[], bool],
    tensor_factory: Callable[[Sequence[float]], Any],
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    on_tick: Optional[Callable[[dict[str, Any]], None]] = None,
    preflight_timeout_sec: float = 0.0,
) -> Optional[HomeSnapshot]:
    """Run hand guiding until save is requested, timeout occurs, or an error is raised.

    The function only terminates a policy after it has successfully started that
    policy itself. A timeout returns ``None`` and never saves a home pose.
    """

    config.validate()
    # Repeat the complete preflight immediately before policy ownership changes.
    # The CLI performs an earlier preflight for operator review, but robot state
    # may have changed while the operator was reading and confirming it.
    wait_for_stationary_preflight(
        robot,
        timeout_sec=preflight_timeout_sec,
        monotonic=monotonic,
        sleep=sleep,
    )

    policy_started = False
    start_time = monotonic()
    period = 1.0 / config.follow_hz
    try:
        stiffness = tensor_factory(config.stiffness)
        damping = tensor_factory(config.damping)
        robot.start_cartesian_impedance(Kx=stiffness, Kxd=damping)
        policy_started = True
        if not robot.is_running_policy():
            raise FreedriveError("Polymetis did not report the low-impedance policy as running")

        while True:
            now = monotonic()
            if now - start_time >= config.max_duration_sec:
                return None

            position, quat = robot.get_ee_pose()
            position_list = _to_list(position)
            quat_list = _to_list(quat)
            update_index = robot.update_desired_ee_pose(
                tensor_factory(position_list),
                tensor_factory(quat_list),
            )
            if update_index == -1:
                raise FreedriveError("Polymetis rejected the current-pose follow update")

            if on_tick is not None:
                on_tick(
                    {
                        "elapsed_sec": now - start_time,
                        "position_m": position_list,
                        "quat_xyzw": _normalise_quaternion(quat_list).tolist(),
                    }
                )
            if should_save():
                return capture_home_snapshot(robot)

            elapsed = monotonic() - now
            sleep(max(0.0, period - elapsed))
    finally:
        if policy_started:
            try:
                robot.terminate_current_policy()
            except Exception as exc:
                raise FreedriveError(f"failed to terminate the low-impedance policy: {exc}") from exc
