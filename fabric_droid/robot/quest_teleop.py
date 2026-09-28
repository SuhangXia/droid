"""Safety-critical, right-hand Quest to Polymetis teleoperation primitives.

The module has no import-time dependency on Polymetis, PyTorch, ADB, or the
Oculus reader.  This keeps preview/tests independent and makes the real robot
control gates explicit in the CLI.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from fabric_droid.robot.gripper_control import GripperControlError, read_gripper
from fabric_droid.robot.freedrive_home import wait_for_stationary_preflight


class QuestTeleopError(RuntimeError):
    """Raised when a Quest or robot safety invariant is violated."""


@dataclass(frozen=True)
class QuestFrame:
    """One newly received Quest right-controller frame."""

    pose: np.ndarray
    buttons: Mapping[str, Any]
    received_monotonic_ns: int
    sequence: int


@dataclass(frozen=True)
class QuestTeleopConfig:
    """Conservative bounds for a first real-robot teleoperation test."""

    control_hz: float = 30.0
    # Zero explicitly disables only the total session timer. All deadman,
    # controller watchdog, robot-state, tracking, speed and workspace limits
    # remain active.
    max_duration_sec: float = 120.0
    controller_timeout_sec: float = 0.25
    max_translation_speed_m_s: float = 0.05
    max_rotation_speed_rad_s: float = math.radians(20.0)
    max_position_tracking_error_m: float = 0.05
    max_rotation_tracking_error_rad: float = math.radians(20.0)
    max_consecutive_command_failures: int = 3
    enforce_workspace_limits: bool = True
    workspace_min_m: tuple[float, float, float] = (0.20, -0.60, 0.05)
    workspace_max_m: tuple[float, float, float] = (0.80, 0.60, 0.90)
    workspace_recovery_margin_m: float = 0.02
    cartesian_stiffness: tuple[float, float, float, float, float, float] = (
        120.0,
        120.0,
        120.0,
        10.0,
        10.0,
        10.0,
    )
    cartesian_damping: tuple[float, float, float, float, float, float] = (
        24.0,
        24.0,
        24.0,
        3.0,
        3.0,
        3.0,
    )

    def validate(self) -> None:
        finite_positive = {
            "control_hz": self.control_hz,
            "controller_timeout_sec": self.controller_timeout_sec,
            "max_translation_speed_m_s": self.max_translation_speed_m_s,
            "max_rotation_speed_rad_s": self.max_rotation_speed_rad_s,
            "max_position_tracking_error_m": self.max_position_tracking_error_m,
            "max_rotation_tracking_error_rad": self.max_rotation_tracking_error_rad,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise QuestTeleopError(f"{name} must be finite and positive")
        if not 10.0 <= self.control_hz <= 60.0:
            raise QuestTeleopError("control_hz must be within [10, 60]")
        if not math.isfinite(self.max_duration_sec) or self.max_duration_sec < 0.0:
            raise QuestTeleopError(
                "max_duration_sec must be finite and non-negative; use 0 for no session timeout"
            )
        if self.max_duration_sec > 600.0:
            raise QuestTeleopError(
                "max_duration_sec must not exceed 600; use exactly 0 for no session timeout"
            )
        if self.controller_timeout_sec > 1.0:
            raise QuestTeleopError("controller_timeout_sec must not exceed 1 second")
        if self.max_translation_speed_m_s > 0.15:
            raise QuestTeleopError("translation speed limit must not exceed 0.15 m/s")
        if self.max_rotation_speed_rad_s > math.radians(60.0):
            raise QuestTeleopError("rotation speed limit must not exceed 60 deg/s")
        if (
            not isinstance(self.max_consecutive_command_failures, int)
            or not 2 <= self.max_consecutive_command_failures <= 10
        ):
            raise QuestTeleopError("max_consecutive_command_failures must be an integer within [2, 10]")
        if (
            not math.isfinite(self.workspace_recovery_margin_m)
            or self.workspace_recovery_margin_m < 0.0
            or self.workspace_recovery_margin_m > 0.05
        ):
            raise QuestTeleopError("workspace_recovery_margin_m must be within [0, 0.05]")

        workspace_min = _vector(self.workspace_min_m, 3, "workspace_min_m")
        workspace_max = _vector(self.workspace_max_m, 3, "workspace_max_m")
        if np.any(workspace_min >= workspace_max):
            raise QuestTeleopError("workspace_min_m must be strictly less than workspace_max_m")

        stiffness = _vector(self.cartesian_stiffness, 6, "cartesian_stiffness")
        damping = _vector(self.cartesian_damping, 6, "cartesian_damping")
        if np.any(stiffness < 0.0) or np.any(damping < 0.0):
            raise QuestTeleopError("Cartesian stiffness/damping must be non-negative")
        if np.any(stiffness[:3] > 250.0) or np.any(stiffness[3:] > 30.0):
            raise QuestTeleopError(
                "refusing high Cartesian stiffness; limits are 250 translation and 30 rotation"
            )
        if np.any(damping[:3] > 50.0) or np.any(damping[3:] > 10.0):
            raise QuestTeleopError(
                "refusing high Cartesian damping; limits are 50 translation and 10 rotation"
            )


@dataclass(frozen=True)
class QuestGripperConfig:
    """DROID-compatible right-trigger position control for one gripper."""

    max_width_m: float
    speed_m_s: float = 0.02
    force_n: float = 10.0
    command_hz: float = 15.0
    state_check_hz: float = 5.0
    min_target_step_m: float = 0.0005
    max_closedness: float = 0.98

    def validate(self) -> None:
        values = {
            "max_width_m": self.max_width_m,
            "speed_m_s": self.speed_m_s,
            "force_n": self.force_n,
            "command_hz": self.command_hz,
            "state_check_hz": self.state_check_hz,
            "min_target_step_m": self.min_target_step_m,
            "max_closedness": self.max_closedness,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise QuestTeleopError(f"{name} must be finite")
        if not 0.01 <= self.max_width_m <= 0.20:
            raise QuestTeleopError("max_width_m must be within [0.01, 0.20]")
        if not 0.001 <= self.speed_m_s <= 0.05:
            raise QuestTeleopError("gripper speed must be within [1, 50] mm/s")
        if not 1.0 <= self.force_n <= 40.0:
            raise QuestTeleopError("gripper force must be within [1, 40] N")
        if not 1.0 <= self.command_hz <= 30.0:
            raise QuestTeleopError("gripper command_hz must be within [1, 30]")
        if not 1.0 <= self.state_check_hz <= 30.0:
            raise QuestTeleopError("gripper state_check_hz must be within [1, 30]")
        if not 0.0001 <= self.min_target_step_m <= 0.01:
            raise QuestTeleopError("gripper min_target_step_m must be within [0.1, 10] mm")
        if not 0.50 <= self.max_closedness <= 1.0:
            raise QuestTeleopError("gripper max_closedness must be within [0.50, 1.0]")

    @property
    def min_width_m(self) -> float:
        return self.max_width_m * (1.0 - self.max_closedness)


@dataclass(frozen=True)
class TeleopDecision:
    """A single safe state-machine decision."""

    send_target: bool
    position_m: np.ndarray
    quat_xyzw: np.ndarray
    deadman_pressed: bool
    workspace_clamped: bool = False
    workspace_recovery_active: bool = False
    exit_requested: bool = False


def _vector(values: Sequence[float], length: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != length:
        raise QuestTeleopError(f"{name} must contain {length} values")
    if not np.all(np.isfinite(array)):
        raise QuestTeleopError(f"{name} contains NaN or infinity")
    return array


def _normalise_quat(quat_xyzw: Sequence[float]) -> np.ndarray:
    quat = _vector(quat_xyzw, 4, "quat_xyzw")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise QuestTeleopError("quaternion norm is near zero")
    return quat / norm


def quat_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalise_quat(quat_xyzw)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a proper rotation matrix to an xyzw quaternion."""

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise QuestTeleopError("rotation matrix must be finite and 3x3")
    # Eigen-decomposition is stable near all trace values and avoids branch
    # singularities.  The matrix below follows Bar-Itzhack's construction.
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    k = np.asarray(
        [
            [m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12],
            [m01 + m10, m11 - m00 - m22, m12 + m21, m02 - m20],
            [m02 + m20, m12 + m21, m22 - m00 - m11, m10 - m01],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(k / 3.0)
    quat = eigenvectors[:, int(np.argmax(eigenvalues))]
    if quat[3] < 0.0:
        quat = -quat
    return _normalise_quat(quat)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if abs(math.pi - angle) < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


def _rotation_from_vector(vector: Sequence[float]) -> np.ndarray:
    rotvec = _vector(vector, 3, "rotation_vector")
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def rotation_error_rad(target_quat: Sequence[float], measured_quat: Sequence[float]) -> float:
    relative = quat_to_matrix(target_quat) @ quat_to_matrix(measured_quat).T
    return float(np.linalg.norm(_rotation_vector(relative)))


def workspace_violation(
    position_m: Sequence[float],
    workspace_min_m: Sequence[float],
    workspace_max_m: Sequence[float],
) -> np.ndarray:
    """Return per-axis distance outside a Cartesian box (zero when inside)."""

    position = _vector(position_m, 3, "position_m")
    workspace_min = _vector(workspace_min_m, 3, "workspace_min_m")
    workspace_max = _vector(workspace_max_m, 3, "workspace_max_m")
    return np.maximum(workspace_min - position, 0.0) + np.maximum(
        position - workspace_max, 0.0
    )


def right_trigger_closedness(buttons: Mapping[str, Any]) -> float:
    """Return the DROID right-trigger closedness in [0, 1]."""

    if "rightTrig" not in buttons:
        raise QuestTeleopError("right-controller frame is missing analog rightTrig")
    raw = buttons["rightTrig"]
    if isinstance(raw, (tuple, list, np.ndarray)):
        if len(raw) < 1:
            raise QuestTeleopError("rightTrig analog value is empty")
        raw = raw[0]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise QuestTeleopError(f"rightTrig is not numeric: {raw!r}") from exc
    if not math.isfinite(value) or not -0.05 <= value <= 1.05:
        raise QuestTeleopError(f"rightTrig value {value!r} is outside the expected [0, 1] range")
    return float(np.clip(value, 0.0, 1.0))


class QuestGripperController:
    """Rate-limited DROID trigger-to-width adapter for Polymetis."""

    def __init__(self, gripper: Any, config: QuestGripperConfig) -> None:
        config.validate()
        self.gripper = gripper
        self.config = config
        self._movement_enabled = False
        self._ever_commanded = False
        self._last_target_width_m: Optional[float] = None
        self._last_command_ns: Optional[int] = None
        self._last_state_check_ns: Optional[int] = None
        self._last_measured_width_m: Optional[float] = None

    def _checked_state(self) -> dict[str, Any]:
        try:
            state = read_gripper(self.gripper)
        except GripperControlError as exc:
            raise QuestTeleopError(f"gripper state failed: {exc}") from exc
        if int(state["error_code"]) != 0:
            raise QuestTeleopError(f"gripper error_code became {state['error_code']}")
        if not bool(state["prev_command_successful"]):
            raise QuestTeleopError("gripper reported an unsuccessful previous command")
        self._last_measured_width_m = float(state["width_m"])
        return state

    def measured_closedness(self, now_ns: int) -> float:
        """Return cached/rate-limited measured closedness using DROID's convention."""

        state_check_period_ns = int(1e9 / self.config.state_check_hz)
        if (
            self._last_measured_width_m is None
            or self._last_state_check_ns is None
            or now_ns - self._last_state_check_ns >= state_check_period_ns
        ):
            self._checked_state()
            self._last_state_check_ns = now_ns
        assert self._last_measured_width_m is not None
        return float(
            np.clip(
                1.0 - self._last_measured_width_m / self.config.max_width_m,
                0.0,
                1.0,
            )
        )

    def hold_current(self) -> Optional[dict[str, Any]]:
        """Freeze an in-flight gripper move at its measured width."""

        if not self._ever_commanded:
            self._movement_enabled = False
            return None
        state = self._checked_state()
        measured_width_m = float(state["width_m"])
        width_m = max(measured_width_m, self.config.min_width_m)
        self.gripper.goto(
            width=width_m,
            speed=self.config.speed_m_s,
            force=self.config.force_n,
            blocking=False,
        )
        self._movement_enabled = False
        self._last_target_width_m = width_m
        return {
            "enabled": False,
            "command_sent": True,
            "closedness": float(
                np.clip(1.0 - (measured_width_m / self.config.max_width_m), 0.0, 1.0)
            ),
            "commanded_closedness": 1.0 - (width_m / self.config.max_width_m),
            "target_width_m": width_m,
            "target_width_mm": width_m * 1000.0,
            "hold_reason": "deadman_or_session_stop",
        }

    def step(
        self,
        frame: QuestFrame,
        *,
        movement_enabled: bool,
        now_ns: int,
    ) -> dict[str, Any]:
        closedness = right_trigger_closedness(frame.buttons)
        commanded_closedness = min(closedness, self.config.max_closedness)
        target_width_m = self.config.max_width_m * (1.0 - commanded_closedness)
        if not movement_enabled:
            if self._movement_enabled:
                held = self.hold_current()
                assert held is not None
                held["closedness"] = closedness
                held["commanded_closedness"] = min(
                    float(held["commanded_closedness"]),
                    self.config.max_closedness,
                )
                return held
            return {
                "enabled": False,
                "command_sent": False,
                "closedness": closedness,
                "commanded_closedness": commanded_closedness,
                "target_width_m": self._last_target_width_m,
                "target_width_mm": (
                    None
                    if self._last_target_width_m is None
                    else self._last_target_width_m * 1000.0
                ),
            }

        self._movement_enabled = True
        command_period_ns = int(1e9 / self.config.command_hz)
        command_due = (
            self._last_command_ns is None
            or now_ns - self._last_command_ns >= command_period_ns
        )
        target_changed = (
            self._last_target_width_m is None
            or abs(target_width_m - self._last_target_width_m)
            >= self.config.min_target_step_m
        )
        command_sent = False
        if command_due and target_changed:
            self.gripper.goto(
                width=target_width_m,
                speed=self.config.speed_m_s,
                force=self.config.force_n,
                blocking=False,
            )
            self._ever_commanded = True
            self._last_target_width_m = target_width_m
            self._last_command_ns = now_ns
            command_sent = True

        state_check_period_ns = int(1e9 / self.config.state_check_hz)
        if (
            self._last_state_check_ns is None
            or now_ns - self._last_state_check_ns >= state_check_period_ns
        ):
            self._checked_state()
            self._last_state_check_ns = now_ns
        return {
            "enabled": True,
            "command_sent": command_sent,
            "closedness": closedness,
            "commanded_closedness": commanded_closedness,
            "target_width_m": target_width_m,
            "target_width_mm": target_width_m * 1000.0,
        }


def apply_workspace_recovery_guard(
    proposed_position_m: Sequence[float],
    previous_position_m: Sequence[float],
    *,
    workspace_min_m: Sequence[float],
    workspace_max_m: Sequence[float],
    recovery_margin_m: float,
) -> tuple[np.ndarray, bool, bool]:
    """Clamp normal targets and allow an outside target to move only inward.

    Returns ``(guarded_position, intervention, recovery_active)``.  An outside
    previous target is never snapped to the boundary; the normal velocity limit
    remains authoritative while outward motion is blocked per axis.
    """

    proposed = _vector(proposed_position_m, 3, "proposed_position_m").copy()
    previous = _vector(previous_position_m, 3, "previous_position_m")
    workspace_min = _vector(workspace_min_m, 3, "workspace_min_m")
    workspace_max = _vector(workspace_max_m, 3, "workspace_max_m")
    previous_violation = workspace_violation(previous, workspace_min, workspace_max)
    if np.any(previous_violation > recovery_margin_m + 1e-9):
        raise QuestTeleopError(
            "flange target is outside the permitted workspace recovery margin: "
            f"violation={previous_violation.tolist()}m, margin={recovery_margin_m:.3f}m"
        )

    intervention = False
    recovery_active = bool(np.any(previous_violation > 0.0))
    for axis in range(3):
        if previous[axis] < workspace_min[axis]:
            # Below the lower boundary: do not let the target decrease further.
            guarded = max(proposed[axis], previous[axis])
            intervention = intervention or not np.isclose(guarded, proposed[axis])
            proposed[axis] = min(guarded, workspace_max[axis])
        elif previous[axis] > workspace_max[axis]:
            # Above the upper boundary: do not let the target increase further.
            guarded = min(proposed[axis], previous[axis])
            intervention = intervention or not np.isclose(guarded, proposed[axis])
            proposed[axis] = max(guarded, workspace_min[axis])
        else:
            guarded = float(np.clip(proposed[axis], workspace_min[axis], workspace_max[axis]))
            intervention = intervention or not np.isclose(guarded, proposed[axis])
            proposed[axis] = guarded
    return proposed, intervention, recovery_active


def validate_quest_frame(frame: QuestFrame, *, now_ns: int, timeout_sec: float) -> None:
    pose = np.asarray(frame.pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise QuestTeleopError("right-controller pose must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise QuestTeleopError("right-controller pose has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-3
    ):
        raise QuestTeleopError("right-controller pose rotation is not a proper rotation matrix")
    age_sec = (int(now_ns) - int(frame.received_monotonic_ns)) / 1e9
    if age_sec < -0.01 or age_sec > timeout_sec:
        raise QuestTeleopError(
            f"Quest controller watchdog expired: newest frame age is {age_sec:.3f}s "
            f"(limit {timeout_sec:.3f}s)"
        )
    for key in ("RG", "RJ", "B"):
        if key not in frame.buttons:
            raise QuestTeleopError(f"right-controller frame is missing button {key}")


class RightQuestTargetMapper:
    """Map right-controller motion into bounded Franka flange targets."""

    # Same handedness-preserving DROID right-controller axis mapping as
    # rmat_reorder=[-2, -1, -3, 4].
    _QUEST_TO_ROBOT = np.asarray(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )

    def __init__(self, config: QuestTeleopConfig) -> None:
        config.validate()
        self.config = config
        self._deadman = False
        self._armed = False
        self._forward_rotation: Optional[np.ndarray] = None
        self._controller_origin: Optional[np.ndarray] = None
        self._robot_origin_position: Optional[np.ndarray] = None
        self._robot_origin_rotation: Optional[np.ndarray] = None
        self._last_target_position: Optional[np.ndarray] = None
        self._last_target_rotation: Optional[np.ndarray] = None

    def _workspace_violation(self, position_m: Sequence[float]) -> np.ndarray:
        if not self.config.enforce_workspace_limits:
            return np.zeros(3, dtype=np.float64)
        return workspace_violation(
            position_m,
            self.config.workspace_min_m,
            self.config.workspace_max_m,
        )

    def _guard_workspace(
        self,
        proposed_position_m: Sequence[float],
        previous_position_m: Sequence[float],
    ) -> tuple[np.ndarray, bool, bool]:
        if not self.config.enforce_workspace_limits:
            return (
                _vector(proposed_position_m, 3, "proposed_position_m").copy(),
                False,
                False,
            )
        return apply_workspace_recovery_guard(
            proposed_position_m,
            previous_position_m,
            workspace_min_m=self.config.workspace_min_m,
            workspace_max_m=self.config.workspace_max_m,
            recovery_margin_m=self.config.workspace_recovery_margin_m,
        )

    def arm(self, frame: QuestFrame, position_m: Sequence[float], quat_xyzw: Sequence[float]) -> None:
        """Arm only while RG is released, preventing motion at controller startup."""

        if bool(frame.buttons["RG"]):
            raise QuestTeleopError("release the right side-grip RG before arming teleoperation")
        pose = np.asarray(frame.pose, dtype=np.float64)
        self._forward_rotation = pose[:3, :3].copy()
        self._last_target_position = _vector(position_m, 3, "ee_position")
        self._last_target_rotation = quat_to_matrix(quat_xyzw)
        self._armed = True

    def force_hold(
        self,
        measured_position_m: Sequence[float],
        measured_quat_xyzw: Sequence[float],
    ) -> TeleopDecision:
        """Immediately freeze at the measured pose and reset the deadman origin."""

        if not self._armed:
            raise QuestTeleopError("teleoperation mapper has not been safely armed")
        measured_position = _vector(measured_position_m, 3, "measured_position_m")
        measured_rotation = quat_to_matrix(measured_quat_xyzw)
        violation = self._workspace_violation(measured_position)
        if np.any(violation > self.config.workspace_recovery_margin_m + 1e-9):
            raise QuestTeleopError(
                "cannot enter command-failure HOLD outside the workspace recovery margin: "
                f"violation={violation.tolist()}m"
            )
        self._deadman = False
        self._controller_origin = None
        self._robot_origin_position = None
        self._robot_origin_rotation = None
        self._last_target_position = measured_position.copy()
        self._last_target_rotation = measured_rotation.copy()
        return TeleopDecision(
            send_target=True,
            position_m=measured_position,
            quat_xyzw=matrix_to_quat(measured_rotation),
            deadman_pressed=False,
            workspace_recovery_active=bool(np.any(violation > 0.0)),
        )

    def step(
        self,
        frame: QuestFrame,
        measured_position_m: Sequence[float],
        measured_quat_xyzw: Sequence[float],
    ) -> TeleopDecision:
        if not self._armed:
            raise QuestTeleopError("teleoperation mapper has not been safely armed")
        measured_position = _vector(measured_position_m, 3, "measured_position_m")
        measured_rotation = quat_to_matrix(measured_quat_xyzw)
        controller_pose = np.asarray(frame.pose, dtype=np.float64)
        deadman = bool(frame.buttons["RG"])
        measured_violation = self._workspace_violation(measured_position)
        if np.any(measured_violation > self.config.workspace_recovery_margin_m + 1e-9):
            raise QuestTeleopError(
                "measured flange pose exceeded the workspace recovery margin: "
                f"violation={measured_violation.tolist()}m, "
                f"margin={self.config.workspace_recovery_margin_m:.3f}m"
            )

        if bool(frame.buttons["B"]):
            return TeleopDecision(
                send_target=False,
                position_m=measured_position,
                quat_xyzw=matrix_to_quat(measured_rotation),
                deadman_pressed=deadman,
                exit_requested=True,
            )

        if bool(frame.buttons["RJ"]) and not deadman:
            self._forward_rotation = controller_pose[:3, :3].copy()

        if not deadman:
            send_target = self._deadman
            if self._last_target_position is None:
                send_target = True
            if send_target:
                # Freeze exactly where the flange is when RG is released.
                self._last_target_position = measured_position.copy()
                self._last_target_rotation = measured_rotation.copy()
                violation = self._workspace_violation(self._last_target_position)
                if np.any(violation > self.config.workspace_recovery_margin_m + 1e-9):
                    raise QuestTeleopError(
                        "measured flange pose on RG release exceeded the workspace recovery "
                        f"margin: violation={violation.tolist()}m"
                    )
            self._deadman = False
            self._controller_origin = None
            recovery_active = bool(
                np.any(
                    self._workspace_violation(self._last_target_position)
                    > 0.0
                )
            )
            return TeleopDecision(
                send_target=send_target,
                position_m=self._last_target_position.copy(),
                quat_xyzw=matrix_to_quat(self._last_target_rotation),
                deadman_pressed=False,
                workspace_recovery_active=recovery_active,
            )

        if not self._deadman:
            self._controller_origin = controller_pose.copy()
            self._robot_origin_position = measured_position.copy()
            self._robot_origin_rotation = measured_rotation.copy()
            self._last_target_position = measured_position.copy()
            self._last_target_rotation = measured_rotation.copy()
            violation = self._workspace_violation(self._last_target_position)
            if np.any(violation > self.config.workspace_recovery_margin_m + 1e-9):
                raise QuestTeleopError(
                    "measured flange pose exceeded the workspace recovery margin while "
                    f"enabling RG: violation={violation.tolist()}m"
                )
            self._deadman = True
            return TeleopDecision(
                send_target=True,
                position_m=measured_position,
                quat_xyzw=matrix_to_quat(measured_rotation),
                deadman_pressed=True,
                workspace_recovery_active=bool(np.any(violation > 0.0)),
            )

        assert self._controller_origin is not None
        assert self._robot_origin_position is not None
        assert self._robot_origin_rotation is not None
        assert self._forward_rotation is not None
        assert self._last_target_position is not None
        assert self._last_target_rotation is not None

        local_translation = self._forward_rotation.T @ (
            controller_pose[:3, 3] - self._controller_origin[:3, 3]
        )
        raw_position = self._robot_origin_position + self._QUEST_TO_ROBOT @ local_translation

        transformed_rotation = (
            self._QUEST_TO_ROBOT @ self._forward_rotation.T @ controller_pose[:3, :3]
        )
        transformed_origin = (
            self._QUEST_TO_ROBOT
            @ self._forward_rotation.T
            @ self._controller_origin[:3, :3]
        )
        relative_rotation = transformed_rotation @ transformed_origin.T
        raw_rotation = relative_rotation @ self._robot_origin_rotation

        max_position_step = self.config.max_translation_speed_m_s / self.config.control_hz
        position_delta = raw_position - self._last_target_position
        position_delta_norm = float(np.linalg.norm(position_delta))
        if position_delta_norm > max_position_step:
            position_delta *= max_position_step / position_delta_norm
        target_position = self._last_target_position + position_delta

        relative_step = raw_rotation @ self._last_target_rotation.T
        rotation_vector = _rotation_vector(relative_step)
        max_rotation_step = self.config.max_rotation_speed_rad_s / self.config.control_hz
        rotation_step_norm = float(np.linalg.norm(rotation_vector))
        if rotation_step_norm > max_rotation_step:
            rotation_vector *= max_rotation_step / rotation_step_norm
        target_rotation = _rotation_from_vector(rotation_vector) @ self._last_target_rotation

        target_position, workspace_clamped, recovery_active = self._guard_workspace(
            target_position,
            self._last_target_position,
        )

        position_error = float(np.linalg.norm(target_position - measured_position))
        target_quat = matrix_to_quat(target_rotation)
        measured_quat = matrix_to_quat(measured_rotation)
        rotation_error = rotation_error_rad(target_quat, measured_quat)
        if position_error > self.config.max_position_tracking_error_m:
            raise QuestTeleopError(
                f"position tracking error {position_error:.3f}m exceeds "
                f"{self.config.max_position_tracking_error_m:.3f}m"
            )
        if rotation_error > self.config.max_rotation_tracking_error_rad:
            raise QuestTeleopError(
                f"rotation tracking error {math.degrees(rotation_error):.1f}deg exceeds "
                f"{math.degrees(self.config.max_rotation_tracking_error_rad):.1f}deg"
            )

        self._last_target_position = target_position.copy()
        self._last_target_rotation = target_rotation.copy()
        return TeleopDecision(
            send_target=True,
            position_m=target_position,
            quat_xyzw=target_quat,
            deadman_pressed=True,
            workspace_clamped=workspace_clamped,
            workspace_recovery_active=recovery_active,
        )


def _to_numpy(value: Any, length: int, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return _vector(value, length, name)


def _terminate_owned_policy(robot: Any) -> None:
    try:
        robot.terminate_current_policy(return_log=False)
    except TypeError:
        robot.terminate_current_policy()


def run_right_quest_teleop(
    robot: Any,
    *,
    frame_source: Any,
    config: QuestTeleopConfig,
    tensor_factory: Callable[[Sequence[float]], Any],
    gripper_controller: Optional[QuestGripperController] = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    on_tick: Optional[Callable[[dict[str, Any]], None]] = None,
    preflight_timeout_sec: float = 0.0,
) -> str:
    """Run a guarded teleoperation session and terminate only our own policy."""

    config.validate()
    preflight = wait_for_stationary_preflight(
        robot,
        timeout_sec=preflight_timeout_sec,
        monotonic=lambda: monotonic_ns() / 1e9,
        sleep=sleep,
    )
    initial_position = _vector(preflight["ee_position_m"], 3, "initial_position")
    initial_quat = _normalise_quat(preflight["ee_quat_xyzw"])
    initial_violation = (
        workspace_violation(
            initial_position,
            config.workspace_min_m,
            config.workspace_max_m,
        )
        if config.enforce_workspace_limits
        else np.zeros(3, dtype=np.float64)
    )
    if np.any(initial_violation > config.workspace_recovery_margin_m + 1e-9):
        raise QuestTeleopError(
            f"initial flange position {initial_position.tolist()} is too far outside the "
            "configured workspace for recovery: "
            f"violation={initial_violation.tolist()}m, "
            f"margin={config.workspace_recovery_margin_m:.3f}m"
        )

    first_frame = frame_source.latest_frame()
    if first_frame is None:
        raise QuestTeleopError("no right-controller frame is available")
    validate_quest_frame(
        first_frame,
        now_ns=monotonic_ns(),
        timeout_sec=config.controller_timeout_sec,
    )
    mapper = RightQuestTargetMapper(config)
    mapper.arm(first_frame, initial_position, initial_quat)

    # Recheck immediately before taking policy ownership.
    wait_for_stationary_preflight(
        robot,
        timeout_sec=preflight_timeout_sec,
        monotonic=lambda: monotonic_ns() / 1e9,
        sleep=sleep,
    )
    policy_started = False
    started_ns = monotonic_ns()
    last_sequence = first_frame.sequence
    command_failure_streak = 0
    command_hold_latched = False
    period = 1.0 / config.control_hz
    try:
        robot.start_cartesian_impedance(
            Kx=tensor_factory(config.cartesian_stiffness),
            Kxd=tensor_factory(config.cartesian_damping),
        )
        policy_started = True
        if not robot.is_running_policy():
            raise QuestTeleopError("Polymetis did not report the teleoperation policy as running")
        update_index = robot.update_desired_ee_pose(
            tensor_factory(initial_position),
            tensor_factory(initial_quat),
        )
        if int(update_index) < 0:
            raise QuestTeleopError("inverse kinematics rejected the initial hold pose")

        while True:
            loop_started_ns = monotonic_ns()
            elapsed_sec = (loop_started_ns - started_ns) / 1e9
            if config.max_duration_sec > 0.0 and elapsed_sec >= config.max_duration_sec:
                return "timeout"
            if not robot.is_running_policy():
                raise QuestTeleopError("Polymetis policy stopped unexpectedly")

            frame = frame_source.latest_frame()
            if frame is None:
                raise QuestTeleopError("right-controller frame disappeared")
            validate_quest_frame(
                frame,
                now_ns=loop_started_ns,
                timeout_sec=config.controller_timeout_sec,
            )
            if frame.sequence < last_sequence:
                raise QuestTeleopError("Quest frame sequence moved backwards")
            last_sequence = frame.sequence

            measured_position_t, measured_quat_t = robot.get_ee_pose()
            measured_position = _to_numpy(measured_position_t, 3, "measured_position")
            measured_quat = _normalise_quat(_to_numpy(measured_quat_t, 4, "measured_quat"))
            actual_deadman = bool(frame.buttons["RG"])
            if command_hold_latched and actual_deadman:
                # A command anomaly may have occurred while moving. Keep the
                # mapper released until the operator physically releases RG,
                # preventing an automatic resume from an old controller origin.
                safe_buttons = dict(frame.buttons)
                safe_buttons["RG"] = False
                effective_frame = QuestFrame(
                    pose=frame.pose,
                    buttons=safe_buttons,
                    received_monotonic_ns=frame.received_monotonic_ns,
                    sequence=frame.sequence,
                )
            else:
                effective_frame = frame
                if command_hold_latched and not actual_deadman:
                    command_hold_latched = False

            decision = mapper.step(effective_frame, measured_position, measured_quat)
            if decision.exit_requested:
                return "b_button"
            update_index = None
            if decision.send_target:
                update_index = robot.update_desired_ee_pose(
                    tensor_factory(decision.position_m),
                    tensor_factory(decision.quat_xyzw),
                )
                if int(update_index) < 0:
                    raise QuestTeleopError("inverse kinematics rejected a bounded Quest target")

            state = robot.get_robot_state()
            error_code = int(getattr(state, "error_code", 0))
            if error_code != 0:
                raise QuestTeleopError(f"robot error_code became {error_code}")
            if not bool(getattr(state, "prev_command_successful", True)):
                command_failure_streak += 1
                command_hold_latched = True
                decision = mapper.force_hold(measured_position, measured_quat)
                update_index = robot.update_desired_ee_pose(
                    tensor_factory(decision.position_m),
                    tensor_factory(decision.quat_xyzw),
                )
                if int(update_index) < 0:
                    raise QuestTeleopError(
                        "inverse kinematics rejected the command-failure HOLD pose"
                    )
                if command_failure_streak >= config.max_consecutive_command_failures:
                    raise QuestTeleopError(
                        "robot reported "
                        f"{command_failure_streak} consecutive unsuccessful commands; "
                        "owned policy terminated"
                    )
            else:
                command_failure_streak = 0
            gripper_status = None
            if gripper_controller is not None:
                gripper_status = gripper_controller.step(
                    frame,
                    movement_enabled=(
                        decision.deadman_pressed and not command_hold_latched
                    ),
                    now_ns=loop_started_ns,
                )
                measured_gripper_position = gripper_controller.measured_closedness(
                    loop_started_ns
                )
                commanded_gripper_position = float(
                    gripper_status["commanded_closedness"]
                )
            else:
                measured_gripper_position = 0.0
                commanded_gripper_position = 0.0
            if on_tick is not None:
                timestamp = getattr(state, "timestamp", None)
                on_tick(
                    {
                        "timestamp_monotonic_ns": loop_started_ns,
                        "robot_timestamp_seconds": int(
                            getattr(timestamp, "seconds", 0)
                        ),
                        "robot_timestamp_nanos": int(
                            getattr(timestamp, "nanos", 0)
                        ),
                        "elapsed_sec": elapsed_sec,
                        "frame_sequence": frame.sequence,
                        "deadman_pressed": decision.deadman_pressed,
                        "workspace_clamped": decision.workspace_clamped,
                        "workspace_recovery_active": decision.workspace_recovery_active,
                        "command_failure_streak": command_failure_streak,
                        "command_hold_latched": command_hold_latched,
                        "gripper": gripper_status,
                        "gripper_position": measured_gripper_position,
                        "commanded_gripper_position": commanded_gripper_position,
                        "joint_positions_rad": _to_numpy(
                            state.joint_positions,
                            7,
                            "joint_positions",
                        ).tolist(),
                        "joint_velocities_rad_s": _to_numpy(
                            state.joint_velocities,
                            7,
                            "joint_velocities",
                        ).tolist(),
                        "measured_position_m": measured_position.tolist(),
                        "measured_quat_xyzw": measured_quat.tolist(),
                        "target_position_m": decision.position_m.tolist(),
                        "target_quat_xyzw": decision.quat_xyzw.tolist(),
                        "update_index": update_index,
                    }
                )
            remaining = period - ((monotonic_ns() - loop_started_ns) / 1e9)
            if remaining > 0.0:
                sleep(remaining)
    finally:
        if gripper_controller is not None:
            try:
                gripper_controller.hold_current()
            except Exception:
                # Never mask the robot-side exception or prevent termination of
                # the Cartesian policy. The CLI reports gripper faults observed
                # synchronously during the loop.
                pass
        if policy_started:
            _terminate_owned_policy(robot)
