"""Versioned, JSON-serializable Fabric-DROID schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "fabric-droid-1.0"
CANONICAL_DESTINATION_TRAY = "target_tray"
CANONICAL_TASK_INSTRUCTION = "Inspect the fabric and place it in the target tray."
EVENT_NAMES = (
    "approach_start",
    "contact_start",
    "pinch_1_contact",
    "pinch_1_release",
    "pinch_2_contact",
    "pinch_2_release",
    "pinch_3_contact",
    "probe_complete",
    "lift_start",
    "tray_arrival",
    "release_time",
    "episode_end",
)
COLLECTION_EVENT_NAMES = (
    "approach_start",
    "contact_start",
    "probe_complete",
    "lift_start",
    "tray_arrival",
    "release_time",
    "episode_end",
)

DestinationTray = Literal["target_tray"]
Split = Literal["train", "validation", "heldout_test"]


@dataclass(frozen=True)
class EventMarker:
    name: str
    timestamp_monotonic_ns: int
    source: Literal["manual", "automatic", "synthetic"] = "manual"
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in EVENT_NAMES:
            raise ValueError(f"unsupported event name: {self.name}")
        if self.timestamp_monotonic_ns <= 0:
            raise ValueError("event timestamp must be a positive monotonic timestamp")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("event confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    task_instruction: str
    destination_tray: DestinationTray
    source_slot: str
    swatch_uid: str
    split: Split
    start_pose_bucket: str
    success: bool
    failure_reason: str
    operator: str
    session_id: str
    robot_id: str
    camera_serials: dict[str, str]
    gelsight_serial: str
    ati_serial: str
    calibration_id: str
    software_git_commits: dict[str, str]
    schema_version: str = SCHEMA_VERSION
    robot_motion_enabled: bool = False

    def __post_init__(self) -> None:
        if self.destination_tray != CANONICAL_DESTINATION_TRAY:
            raise ValueError(
                "destination_tray must be 'target_tray'; color/direction labels "
                "are no longer part of this task"
            )
        if "target tray" not in self.task_instruction.lower():
            raise ValueError("task_instruction must mention 'target tray'")
        if not self.episode_id or not self.swatch_uid or not self.session_id:
            raise ValueError("episode_id, swatch_uid, and session_id are required")
        if self.success and self.failure_reason:
            raise ValueError("successful episodes cannot have a failure_reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForceCalibration:
    calibration_id: str
    T_ati_to_gripper: list[list[float]] | None = None
    gripper_normal_axis: list[float] | None = None
    force_sign: float | None = None
    bias_wrench: list[float] = field(default_factory=lambda: [0.0] * 6)

    @property
    def normal_force_calibrated(self) -> bool:
        return (
            self.T_ati_to_gripper is not None and self.gripper_normal_axis is not None and self.force_sign in (-1.0, 1.0)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "normal_force_calibrated": self.normal_force_calibrated,
            "warning": None
            if self.normal_force_calibrated
            else "Raw wrench only; normal-force feedback is forbidden without axis calibration.",
        }
