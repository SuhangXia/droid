"""Preflight and force/torque safety gates.

These checks do not replace the Franka hardware safety controller. They prevent
the data-collection process from issuing further close commands after a limit.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fabric_droid.schemas import ForceCalibration


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetyDecision:
    level: str
    allow_further_closing: bool
    abort_episode: bool
    reason: str = ""


@dataclass(frozen=True)
class ForceSafetyGate:
    warning_force_n: float = 12.0
    hard_stop_force_n: float = 20.0
    warning_torque_nm: float = 0.7
    hard_stop_torque_nm: float = 1.0

    def evaluate_raw(self, wrench: Iterable[float]) -> SafetyDecision:
        values = np.asarray(tuple(wrench), dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            return SafetyDecision("hard_stop", False, True, "invalid ATI wrench")
        force_norm = float(np.linalg.norm(values[:3]))
        torque_norm = float(np.linalg.norm(values[3:]))
        if force_norm >= self.hard_stop_force_n or torque_norm >= self.hard_stop_torque_nm:
            return SafetyDecision("hard_stop", False, True, "raw force/torque hard limit exceeded")
        if force_norm >= self.warning_force_n or torque_norm >= self.warning_torque_nm:
            return SafetyDecision("warning", False, False, "raw force/torque warning limit exceeded")
        return SafetyDecision("ok", True, False)

    def normal_force(self, wrench: Iterable[float], calibration: ForceCalibration) -> float:
        if not calibration.normal_force_calibrated:
            raise PreflightError("normal-force control requires T_ati_to_gripper, gripper_normal_axis, and force_sign")
        values = np.asarray(tuple(wrench), dtype=np.float64)
        transform = np.asarray(calibration.T_ati_to_gripper, dtype=np.float64)
        axis = np.asarray(calibration.gripper_normal_axis, dtype=np.float64)
        if transform.shape != (4, 4) or axis.shape != (3,):
            raise PreflightError("invalid ATI calibration dimensions")
        rotated_force = transform[:3, :3] @ (values[:3] - np.asarray(calibration.bias_wrench[:3]))
        return float(calibration.force_sign * np.dot(rotated_force, axis / np.linalg.norm(axis)))


def preflight_output(output_dir: Path, minimum_free_gib: float = 1.0) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".fabric_droid_write_probe"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise PreflightError(f"output directory is not writable: {output_dir}") from exc
    usage = shutil.disk_usage(output_dir)
    free_gib = usage.free / 1024**3
    if free_gib < minimum_free_gib:
        raise PreflightError(f"only {free_gib:.2f} GiB free; require at least {minimum_free_gib:.2f} GiB")
    return {"output_dir": str(output_dir), "free_gib": free_gib, "writable": True}


def assert_no_robot_motion(robot_disabled: bool, record_only: bool, dry_run: bool) -> None:
    if not (robot_disabled or record_only or dry_run):
        raise PreflightError(
            "robot motion is disabled by default; use DROID teleoperation with the explicit sidecar hook "
            "after completing the real-robot pilot checklist"
        )


def validate_robot_preflight(status: dict[str, Any]) -> dict[str, bool]:
    """Validate status supplied by the sole DROID robot-control owner.

    The sidecar never opens a second Franka control connection. The DROID
    process must supply these booleans from its existing robot interface.
    """

    required = (
        "emergency_stop_clear",
        "controllable_mode",
        "joint_limits_ok",
        "workspace_limits_ok",
        "gripper_limits_ok",
    )
    missing = [name for name in required if name not in status]
    failed = [name for name in required if name in status and not bool(status[name])]
    if missing or failed:
        raise PreflightError(f"robot preflight failed; missing={missing}, false={failed}")
    return {name: True for name in required}
