from __future__ import annotations

from typing import Mapping

from fabric_droid_curator.config import SegmentOverlapConfig
from fabric_droid_curator.constants import INSTRUCTION_TEMPLATES
from fabric_droid_curator.schemas import AnnotationPayload, SegmentOut


def clamp_range(start_ns: int, end_ns: int, episode_start_ns: int, episode_end_ns: int) -> tuple[int, int, list[str]]:
    warnings: list[str] = []
    if start_ns < episode_start_ns:
        start_ns = episode_start_ns
        warnings.append("segment_start_clamped_to_episode")
    if end_ns > episode_end_ns:
        end_ns = episode_end_ns
        warnings.append("segment_end_clamped_to_episode")
    if end_ns <= start_ns:
        raise ValueError("segment has no positive duration after timestamp clamp")
    return start_ns, end_ns, warnings


def _segment(
    *,
    annotation: AnnotationPayload,
    annotation_version: int,
    segment_type: str,
    raw_start_ns: int,
    raw_end_ns: int,
    episode_start_ns: int,
    episode_end_ns: int,
) -> SegmentOut:
    start_ns, end_ns, warnings = clamp_range(raw_start_ns, raw_end_ns, episode_start_ns, episode_end_ns)
    template = INSTRUCTION_TEMPLATES[segment_type]
    events = annotation.events
    stable = events.get("stable_grasp")
    split = events.get("branch_point") or stable
    contact = events.get("contact_start")
    tactile_end = stable.timestamp_ns if stable else None
    tactile_start = contact.timestamp_ns if contact and stable else (tactile_end - 500_000_000 if tactile_end else None)
    if tactile_start is not None:
        tactile_start = max(episode_start_ns, tactile_start)
    return SegmentOut(
        segment_id=f"{annotation.episode_id}__{segment_type}__v{annotation_version:03d}",
        source_episode_id=annotation.episode_id,
        segment_type=segment_type,
        start_ns=start_ns,
        end_ns=end_ns,
        duration=(end_ns - start_ns) / 1e9,
        instruction_template_id=template["id"],
        prompt=template["prompt"],
        dataset_decision=annotation.dataset_decision,
        split_point_ns=split.timestamp_ns if split else None,
        contact_start_ns=contact.timestamp_ns if contact else None,
        stable_grasp_ns=stable.timestamp_ns if stable else None,
        tactile_window_start_ns=tactile_start,
        tactile_window_end_ns=tactile_end,
        warnings=warnings,
    )


def build_segments(
    annotation: AnnotationPayload,
    *,
    annotation_version: int,
    episode_start_ns: int,
    episode_end_ns: int,
    overlap: SegmentOverlapConfig,
    allow_draft: bool = False,
) -> list[SegmentOut]:
    if annotation.dataset_decision == "drop":
        return []
    if not allow_draft and annotation.dataset_decision != "keep":
        return []
    if not allow_draft and annotation.review_status != "recovery" and annotation.action_branch == "unknown":
        return []
    events = annotation.events
    trim = events.get("trim_start")
    logical_start_ns = max(episode_start_ns, min(trim.timestamp_ns, episode_end_ns - 1)) if trim else episode_start_ns

    segments: list[SegmentOut] = []
    if annotation.review_status == "recovery":
        segments.append(
            _segment(
                annotation=annotation,
                annotation_version=annotation_version,
                segment_type="recovery",
                raw_start_ns=logical_start_ns,
                raw_end_ns=episode_end_ns,
                episode_start_ns=logical_start_ns,
                episode_end_ns=episode_end_ns,
            )
        )
        return segments

    split = events.get("branch_point") or events.get("stable_grasp")
    if split:
        segments.append(
            _segment(
                annotation=annotation,
                annotation_version=annotation_version,
                segment_type="grasp_probe",
                raw_start_ns=logical_start_ns,
                raw_end_ns=split.timestamp_ns,
                episode_start_ns=logical_start_ns,
                episode_end_ns=episode_end_ns,
            )
        )
    remove_like = annotation.action_branch in {"remove", "legacy_green_left", "legacy_white_right"}
    if remove_like and split:
        segments.append(
            _segment(
                annotation=annotation,
                annotation_version=annotation_version,
                segment_type="remove_to_basket",
                raw_start_ns=split.timestamp_ns,
                raw_end_ns=episode_end_ns,
                episode_start_ns=logical_start_ns,
                episode_end_ns=episode_end_ns,
            )
        )
    if annotation.action_branch == "leave" and split:
        segments.append(
            _segment(
                annotation=annotation,
                annotation_version=annotation_version,
                segment_type="leave_on_rack",
                raw_start_ns=split.timestamp_ns,
                raw_end_ns=episode_end_ns,
                episode_start_ns=logical_start_ns,
                episode_end_ns=episode_end_ns,
            )
        )
    return segments


def event_order_is_valid(events: Mapping[str, int]) -> bool:
    ordered = [
        events[name]
        for name in (
            "motion_start",
            "contact_start",
            "stable_grasp",
            "lift_start",
            "detach_complete",
            "release_start",
            "release_complete",
            "retreat_complete",
        )
        if name in events
    ]
    return all(right > left for left, right in zip(ordered, ordered[1:]))
