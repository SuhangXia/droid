"""Validated joint-space return to a Fabric-DROID saved Franka home."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from fabric_droid.robot.freedrive_home import FreedriveError, preflight_robot


class GoHomeError(RuntimeError):
    """Raised when loading or executing a guarded home motion fails."""


@dataclass(frozen=True)
class SavedHome:
    joint_positions_rad: tuple[float, float, float, float, float, float, float]
    robot_ip: str
    position_m: tuple[float, float, float]
    source_path: Path


def _vector(values: Sequence[float], length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != length or not np.all(np.isfinite(array)):
        raise GoHomeError(f"{name} must contain {length} finite values")
    return array


def load_saved_home(path: Path, *, expected_robot_ip: str) -> SavedHome:
    path = Path(path)
    if not path.is_file():
        raise GoHomeError(f"saved home file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GoHomeError(f"cannot read saved home JSON {path}: {exc}") from exc
    if payload.get("name") != "franka_home_pose" or payload.get("frame") != "franka_base":
        raise GoHomeError("home JSON is not a franka_home_pose in franka_base")
    robot_ip = str(payload.get("robot_ip", ""))
    if robot_ip != expected_robot_ip:
        raise GoHomeError(
            f"home was captured for robot_ip={robot_ip!r}, not {expected_robot_ip!r}"
        )
    joints = _vector(payload.get("joint_positions_rad", []), 7, "joint_positions_rad")
    position = _vector(payload.get("position", []), 3, "position")
    return SavedHome(
        joint_positions_rad=tuple(float(value) for value in joints),
        robot_ip=robot_ip,
        position_m=tuple(float(value) for value in position),
        source_path=path.resolve(),
    )


def home_preflight(robot: Any, home: SavedHome) -> dict[str, Any]:
    state = preflight_robot(robot)
    current = _vector(state["joint_positions_rad"], 7, "current_joint_positions")
    target = _vector(home.joint_positions_rad, 7, "home_joint_positions")
    delta = target - current
    return {
        **state,
        "home_joint_positions_rad": target.tolist(),
        "joint_delta_rad": delta.tolist(),
        "max_abs_joint_delta_rad": float(np.max(np.abs(delta))),
        "joint_delta_norm_rad": float(np.linalg.norm(delta)),
        "home_position_m": list(home.position_m),
        "home_file": str(home.source_path),
    }


def _terminate_owned_policy(robot: Any) -> None:
    try:
        robot.terminate_current_policy(return_log=False)
    except TypeError:
        robot.terminate_current_policy()


def run_go_home(
    robot: Any,
    home: SavedHome,
    *,
    time_to_go_sec: float,
    tensor_factory: Callable[[Sequence[float]], Any],
    tolerance_rad: float = 0.03,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute a finite minimum-jerk joint trajectory to a saved home."""

    if not math.isfinite(time_to_go_sec) or not 3.0 <= time_to_go_sec <= 60.0:
        raise GoHomeError("time_to_go_sec must be within [3, 60]")
    if not math.isfinite(tolerance_rad) or not 0.005 <= tolerance_rad <= 0.10:
        raise GoHomeError("tolerance_rad must be within [0.005, 0.10]")
    try:
        before = home_preflight(robot, home)
    except FreedriveError as exc:
        raise GoHomeError(str(exc)) from exc
    if float(before["max_abs_joint_delta_rad"]) <= tolerance_rad:
        return {"result": "already_home", "before": before, "after": before}

    # Repeat immediately before taking ownership of the controller.
    try:
        home_preflight(robot, home)
    except FreedriveError as exc:
        raise GoHomeError(str(exc)) from exc
    policy_started = False
    started_at = monotonic()
    unsuccessful_streak = 0
    try:
        robot.move_to_joint_positions(
            tensor_factory(home.joint_positions_rad),
            time_to_go=time_to_go_sec,
            blocking=False,
        )
        policy_started = True
        if not robot.is_running_policy():
            raise GoHomeError("Polymetis did not report the home trajectory as running")

        while robot.is_running_policy():
            if monotonic() - started_at > time_to_go_sec + 5.0:
                raise GoHomeError("home trajectory exceeded its time limit")
            state = robot.get_robot_state()
            error_code = int(getattr(state, "error_code", 0))
            if error_code != 0:
                raise GoHomeError(f"robot error_code became {error_code}")
            if bool(getattr(state, "prev_command_successful", True)):
                unsuccessful_streak = 0
            else:
                unsuccessful_streak += 1
                if unsuccessful_streak >= 3:
                    raise GoHomeError("robot reported 3 consecutive unsuccessful commands")
            sleep(0.05)

        final_state = robot.get_robot_state()
        final_joints = _vector(final_state.joint_positions, 7, "final_joint_positions")
        target = _vector(home.joint_positions_rad, 7, "home_joint_positions")
        error = target - final_joints
        max_error = float(np.max(np.abs(error)))
        if int(getattr(final_state, "error_code", 0)) != 0:
            raise GoHomeError(f"robot finished with error_code={int(final_state.error_code)}")
        if not bool(getattr(final_state, "prev_command_successful", True)):
            raise GoHomeError("robot finished with an unsuccessful previous command")
        if max_error > tolerance_rad:
            raise GoHomeError(
                f"home trajectory ended with max joint error {max_error:.4f}rad "
                f"(limit {tolerance_rad:.4f}rad)"
            )
        return {
            "result": "home_reached",
            "before": before,
            "after": {
                "joint_positions_rad": final_joints.tolist(),
                "joint_error_rad": error.tolist(),
                "max_abs_joint_error_rad": max_error,
            },
            "time_to_go_sec": time_to_go_sec,
        }
    finally:
        if policy_started and robot.is_running_policy():
            _terminate_owned_policy(robot)
