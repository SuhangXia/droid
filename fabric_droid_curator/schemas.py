from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import EVENT_ORDER

ActionBranch = Literal[
    "remove",
    "leave",
    "legacy_green_left",
    "legacy_white_right",
    "unknown",
]
ReviewStatus = Literal["unreviewed", "auto_proposed", "accepted", "rejected", "recovery", "verified"]
DatasetDecision = Literal["keep", "drop", "undecided"]


class EventValue(BaseModel):
    timestamp_ns: int = Field(gt=0)
    source: Literal["automatic", "manual", "accepted_automatic"] = "manual"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    detector_version: str | None = None


class MaterialLabels(BaseModel):
    summer_suitability: str | None = None
    softness: str | None = None
    thickness: str | None = None
    breathability: str | None = None
    semantic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    label_source: str | None = None


class AnnotationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    session_id: str | None = None
    swatch_uid: str | None = None
    source_slot: str | None = None
    operator: str | None = None
    camera_session_id: str | None = None
    success: bool | None = None
    review_status: ReviewStatus = "unreviewed"
    failure_reason: str = ""
    single_layer: bool | None = None
    correct_grasp_region: bool | None = None
    stable_hold_present: bool | None = None
    sensor_complete: bool | None = None
    action_branch: ActionBranch = "unknown"
    dataset_decision: DatasetDecision = "undecided"
    events: dict[str, EventValue] = Field(default_factory=dict)
    material_labels: MaterialLabels = Field(default_factory=MaterialLabels)
    notes: str = ""

    @model_validator(mode="after")
    def validate_event_order(self) -> "AnnotationPayload":
        known = [(name, self.events[name].timestamp_ns) for name in EVENT_ORDER if name in self.events]
        if any(right[1] <= left[1] for left, right in zip(known, known[1:])):
            raise ValueError("events must follow strict task order")
        if self.review_status in {"accepted", "verified"} and self.action_branch == "unknown":
            raise ValueError("accepted/verified annotations require an action branch")
        return self


class SaveAnnotationRequest(BaseModel):
    annotation: AnnotationPayload
    reviewer: str
    reason: str = ""
    expected_version: int | None = Field(default=None, ge=0)


class EventProposalOut(BaseModel):
    event_name: str
    candidate_timestamp_ns: int
    confidence: float
    evidence: dict[str, Any]
    detector_version: str


class SegmentOut(BaseModel):
    segment_id: str
    source_episode_id: str
    segment_type: str
    start_ns: int
    end_ns: int
    duration: float
    instruction_template_id: str
    prompt: str
    dataset_decision: DatasetDecision = "undecided"
    split_point_ns: int | None = None
    contact_start_ns: int | None = None
    stable_grasp_ns: int | None = None
    tactile_window_start_ns: int | None = None
    tactile_window_end_ns: int | None = None
    warnings: list[str] = Field(default_factory=list)


class DashboardOut(BaseModel):
    totals: dict[str, int | float]
    review_status: dict[str, int]
    action_branches: dict[str, int]
    swatches: dict[str, int]
    camera_sessions: dict[str, int]
    start_pose_buckets: dict[str, int]
    splits: dict[str, int]
    distributions: dict[str, list[float]]
    matrices: dict[str, list[dict[str, Any]]]
