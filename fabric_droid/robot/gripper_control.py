"""Validated, dependency-light helpers for Polymetis gripper control."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


class GripperControlError(RuntimeError):
    """Raised when a gripper command or state violates a safety invariant."""


MAX_SAFE_CLOSEDNESS = 0.98


def width_for_closedness(max_width_m: float, closedness: float) -> float:
    """Convert a [0, 1] closedness fraction to total finger opening."""

    max_width_m = float(max_width_m)
    closedness = float(closedness)
    if not 0.0 < max_width_m <= 0.20:
        raise GripperControlError(f"invalid gripper max_width_m={max_width_m}")
    if not 0.0 <= closedness <= MAX_SAFE_CLOSEDNESS:
        raise GripperControlError(
            f"closedness {closedness * 100:.2f}% exceeds the hard "
            f"{MAX_SAFE_CLOSEDNESS * 100:.0f}% safety limit"
        )
    return max_width_m * (1.0 - closedness)


def closedness_for_width(max_width_m: float, width_m: float) -> float:
    """Return the closedness fraction represented by a measured width."""

    if not 0.0 < float(max_width_m) <= 0.20:
        raise GripperControlError(f"invalid gripper max_width_m={max_width_m}")
    return 1.0 - float(width_m) / float(max_width_m)


@dataclass(frozen=True)
class GripperMotion:
    """One bounded gripper width command, expressed in SI units."""

    width_m: float
    speed_m_s: float = 0.01
    force_n: float = 5.0

    def validate(
        self,
        *,
        max_width_m: float,
        max_safe_closedness: float = MAX_SAFE_CLOSEDNESS,
    ) -> None:
        values = {
            "width_m": self.width_m,
            "speed_m_s": self.speed_m_s,
            "force_n": self.force_n,
            "max_width_m": max_width_m,
        }
        for name, value in values.items():
            if not isinstance(value, (float, int)) or not float("-inf") < float(value) < float("inf"):
                raise GripperControlError(f"{name} must be finite")
        if max_width_m <= 0.0:
            raise GripperControlError(f"invalid gripper max_width_m={max_width_m}")
        if not 0.0 <= self.width_m <= max_width_m:
            raise GripperControlError(
                f"width {self.width_m * 1000:.2f}mm is outside [0, {max_width_m * 1000:.2f}]mm"
            )
        if not 0.0 < max_safe_closedness <= MAX_SAFE_CLOSEDNESS:
            raise GripperControlError(
                f"max_safe_closedness must be within (0, {MAX_SAFE_CLOSEDNESS}]"
            )
        minimum_safe_width_m = max_width_m * (1.0 - max_safe_closedness)
        if self.width_m < minimum_safe_width_m - 1e-9:
            raise GripperControlError(
                f"width {self.width_m * 1000:.2f}mm would close the gripper to "
                f"{closedness_for_width(max_width_m, self.width_m) * 100:.2f}%; "
                f"the hard safety limit is {max_safe_closedness * 100:.0f}% "
                f"({minimum_safe_width_m * 1000:.2f}mm minimum opening)"
            )
        if not 0.001 <= self.speed_m_s <= 0.05:
            raise GripperControlError("speed must be within [1, 50] mm/s")
        if not 1.0 <= self.force_n <= 40.0:
            raise GripperControlError("force must be within [1, 40] N")


def read_gripper(gripper: Any) -> dict[str, Any]:
    """Return validated metadata and current state without sending a command."""

    metadata = getattr(gripper, "metadata", None)
    if metadata is None:
        raise GripperControlError("gripper metadata is unavailable")
    max_width_m = float(getattr(metadata, "max_width", 0.0))
    if not 0.0 < max_width_m <= 0.20:
        raise GripperControlError(f"invalid max gripper width {max_width_m}")
    state = gripper.get_state()
    width_m = float(getattr(state, "width", -1.0))
    if not 0.0 <= width_m <= max_width_m + 0.005:
        raise GripperControlError(
            f"reported width {width_m:.6f}m is inconsistent with max width {max_width_m:.6f}m"
        )
    timestamp = getattr(state, "timestamp", None)
    return {
        "hz": int(getattr(metadata, "hz", 0)),
        "max_width_m": max_width_m,
        "max_width_mm": max_width_m * 1000.0,
        "width_m": width_m,
        "width_mm": width_m * 1000.0,
        "is_grasped": bool(getattr(state, "is_grasped", False)),
        "is_moving": bool(getattr(state, "is_moving", False)),
        "prev_command_successful": bool(getattr(state, "prev_command_successful", True)),
        "error_code": int(getattr(state, "error_code", 0)),
        "timestamp_seconds": getattr(timestamp, "seconds", None),
        "timestamp_nanos": getattr(timestamp, "nanos", None),
    }


def wait_for_gripper_width(
    gripper: Any,
    *,
    target_width_m: float,
    timeout_s: float = 10.0,
    tolerance_m: float = 0.001,
    poll_interval_s: float = 0.05,
) -> dict[str, Any]:
    """Poll measured state until a commanded width is physically reached."""

    if timeout_s <= 0.0 or tolerance_m <= 0.0 or poll_interval_s <= 0.0:
        raise GripperControlError("width wait timing and tolerance must be positive")
    deadline = time.monotonic() + float(timeout_s)
    last = read_gripper(gripper)
    while True:
        if last["error_code"] != 0:
            raise GripperControlError(f"gripper error_code became {last['error_code']}")
        if (
            abs(float(last["width_m"]) - float(target_width_m)) <= tolerance_m
            and not bool(last["is_moving"])
        ):
            return last
        if time.monotonic() >= deadline:
            raise GripperControlError(
                "gripper did not reach the requested opening: "
                f"target={target_width_m * 1000:.2f}mm, "
                f"measured={float(last['width_m']) * 1000:.2f}mm, "
                f"tolerance={tolerance_m * 1000:.2f}mm"
            )
        time.sleep(poll_interval_s)
        last = read_gripper(gripper)


def move_gripper(
    gripper: Any,
    motion: GripperMotion,
    *,
    verify_width: bool = False,
    blocking: bool = True,
    timeout_s: float = 10.0,
    tolerance_m: float = 0.001,
) -> dict[str, Any]:
    """Execute one bounded width command and return the immediately known state."""

    before = read_gripper(gripper)
    if before["error_code"] != 0:
        raise GripperControlError(f"gripper error_code is {before['error_code']}")
    if not before["prev_command_successful"]:
        raise GripperControlError("previous gripper command was not successful")
    motion.validate(max_width_m=float(before["max_width_m"]))
    gripper.goto(
        width=float(motion.width_m),
        speed=float(motion.speed_m_s),
        force=float(motion.force_n),
        blocking=bool(blocking),
    )
    if not blocking:
        # The gripper server owns the physical motion after a non-blocking
        # goto.  Reading state immediately here races its control loop and
        # can expose the preceding command's status bit as a false failure.
        # The target itself was already validated against the pre-command
        # metadata above, so report accepted submission without a completion
        # assertion. Closing collection paths remain blocking.
        return {"before": before, "after": before, "pending": True}
    after = (
        wait_for_gripper_width(
            gripper,
            target_width_m=motion.width_m,
            timeout_s=timeout_s,
            tolerance_m=tolerance_m,
        )
        if verify_width
        else read_gripper(gripper)
    )
    if after["error_code"] != 0:
        raise GripperControlError(f"gripper error_code became {after['error_code']}")
    if not after["prev_command_successful"]:
        raise GripperControlError("gripper reported an unsuccessful width command")
    return {"before": before, "after": after}
