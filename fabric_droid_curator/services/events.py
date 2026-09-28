from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fabric_droid_curator.config import DetectorConfig
from fabric_droid_curator.constants import EVENT_ORDER

from .signals import SignalBundle

JOINT_TRIM_VERSION = "joint-angle-trim-v1"


@dataclass(frozen=True)
class EventProposalValue:
    event_name: str
    candidate_timestamp_ns: int
    confidence: float
    evidence: dict[str, Any]
    detector_version: str


def detect_joint_angle_trim(
    timestamps_ns: np.ndarray,
    joint_positions: np.ndarray,
    *,
    pre_roll_seconds: float = 0.20,
) -> EventProposalValue:
    """Find the first sustained joint-angle departure and retain a short pre-roll."""
    timestamps = np.asarray(timestamps_ns, dtype=np.int64).reshape(-1)
    positions = np.asarray(joint_positions, dtype=np.float64)
    count = min(timestamps.size, positions.shape[0] if positions.ndim == 2 else 0)
    if count < 4:
        timestamp = int(timestamps[0]) if timestamps.size else 1
        return EventProposalValue(
            event_name="trim_start",
            candidate_timestamp_ns=timestamp,
            confidence=0.1,
            evidence={"method": "insufficient_joint_samples", "sample_count": count},
            detector_version=JOINT_TRIM_VERSION,
        )

    timestamps = timestamps[:count]
    positions = positions[:count]
    baseline = np.median(positions[: min(10, count)], axis=0)
    departure = np.linalg.norm(positions - baseline, axis=1)
    baseline_count = min(count, max(10, round(count * 0.03)))
    baseline_values = departure[:baseline_count]
    baseline_median = float(np.median(baseline_values))
    baseline_mad = float(np.median(np.abs(baseline_values - baseline_median)))
    threshold = max(0.002, baseline_median + 8.0 * max(baseline_mad, 1e-5))
    moving = departure > threshold
    onset_index = _first_sustained(moving, 0, 3)
    found = onset_index is not None
    onset_index = onset_index if onset_index is not None else 0
    median_period_ns = int(np.median(np.diff(timestamps)))
    pre_roll_frames = max(0, round(pre_roll_seconds * 1e9 / max(median_period_ns, 1)))
    trim_index = max(0, onset_index - pre_roll_frames)
    peak_ratio = float(np.max(departure) / max(threshold, 1e-9))
    confidence = float(np.clip(0.55 + 0.1 * peak_ratio, 0.1, 0.98)) if found else 0.1
    return EventProposalValue(
        event_name="trim_start",
        candidate_timestamp_ns=int(timestamps[trim_index]),
        confidence=round(confidence, 4),
        evidence={
            "method": "sustained_joint_angle_departure",
            "joint_departure_threshold_rad": threshold,
            "baseline_median_rad": baseline_median,
            "baseline_mad_rad": baseline_mad,
            "motion_onset_timestamp_ns": int(timestamps[onset_index]),
            "pre_roll_seconds": pre_roll_seconds,
            "sustained_frames": 3,
            "motion_found": found,
        },
        detector_version=JOINT_TRIM_VERSION,
    )


def _robust_score(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    low = float(np.nanpercentile(values, 20))
    high = float(np.nanpercentile(values, 95))
    if high - low < 1e-9:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _interp(target_ns: np.ndarray, source_ns: np.ndarray, values: np.ndarray) -> np.ndarray:
    if not source_ns.size or not values.size:
        return np.zeros(target_ns.shape, dtype=np.float64)
    return np.interp(
        target_ns.astype(np.float64), source_ns.astype(np.float64), values, left=values[0], right=values[-1]
    )


def _first_sustained(mask: np.ndarray, start: int, count: int, end: int | None = None) -> int | None:
    stop = mask.size if end is None else min(end, mask.size)
    run = 0
    for index in range(max(0, start), stop):
        run = run + 1 if mask[index] else 0
        if run >= count:
            return index - count + 1
    return None


def _index_after(timestamps: np.ndarray, timestamp_ns: int) -> int:
    return min(int(np.searchsorted(timestamps, timestamp_ns, side="left")), max(0, timestamps.size - 1))


def detect_events(bundle: SignalBundle, config: DetectorConfig) -> list[EventProposalValue]:
    ts = bundle.robot_timestamp_ns
    if ts.size < 3:
        return []
    dt = float(np.median(np.diff(ts)) / 1e9)

    def count_for(seconds: float) -> int:
        return max(1, round(seconds / max(dt, 1e-3)))

    motion_score = np.maximum(_robust_score(bundle.joint_velocity_norm), _robust_score(bundle.eef_linear_velocity))
    motion_mask = motion_score > 0.18
    motion_index = _first_sustained(motion_mask, 0, count_for(config.motion_min_seconds))
    motion_index = motion_index if motion_index is not None else max(0, int(ts.size * 0.03))

    closing = _robust_score(np.clip(bundle.gripper_velocity, 0.0, None))
    opening = _robust_score(np.clip(-bundle.gripper_velocity, 0.0, None))
    force_score = _interp(ts, bundle.ati_timestamp_ns, _robust_score(bundle.ati_force_delta))
    force_derivative = _interp(ts, bundle.ati_timestamp_ns, _robust_score(bundle.ati_derivative))
    gel_score = _interp(ts, bundle.gelsight_timestamp_ns, _robust_score(bundle.gelsight_contact_score))
    gel_delta = _interp(ts, bundle.gelsight_timestamp_ns, _robust_score(bundle.gelsight_image_delta))
    contact_fusion = 0.34 * closing + 0.28 * force_score + 0.28 * gel_score + 0.10 * gel_delta

    likely_release = _first_sustained(opening > 0.25, motion_index + 1, count_for(config.release_min_seconds))
    release_search_end = likely_release if likely_release is not None else int(ts.size * 0.90)
    contact_index = _first_sustained(
        contact_fusion > 0.28,
        motion_index + 1,
        count_for(config.contact_min_seconds),
        release_search_end,
    )
    if contact_index is None:
        search = contact_fusion[motion_index:release_search_end]
        contact_index = motion_index + int(np.argmax(search)) if search.size else min(ts.size - 1, motion_index + 1)

    stable_mask = (
        (np.abs(bundle.gripper_velocity) < max(0.08, float(np.nanpercentile(np.abs(bundle.gripper_velocity), 55))))
        & (force_derivative < 0.55)
        & (gel_score > 0.12)
    )
    stable_index = _first_sustained(
        stable_mask,
        contact_index + count_for(config.stable_hold_min_seconds),
        count_for(config.stable_hold_min_seconds),
        release_search_end,
    )
    if stable_index is None:
        stable_index = min(release_search_end - 1, contact_index + count_for(max(0.35, config.stable_hold_min_seconds)))

    movement_after_grasp = np.maximum(
        _robust_score(bundle.eef_linear_velocity),
        _robust_score(bundle.joint_velocity_norm),
    )
    lift_index = _first_sustained(
        movement_after_grasp > 0.28,
        stable_index + 1,
        count_for(0.18),
        release_search_end,
    )
    if lift_index is None:
        lift_index = min(release_search_end - 1, stable_index + count_for(0.45))

    release_index = likely_release
    if release_index is None or release_index <= lift_index:
        candidate = opening[lift_index + 1 :]
        release_index = lift_index + 1 + int(np.argmax(candidate)) if candidate.size else int(ts.size * 0.80)
    release_index = min(max(release_index, lift_index + 2), ts.size - 2)

    detach_start = min(release_index - 1, lift_index + 1)
    detach_slice = force_score[detach_start:release_index]
    if detach_slice.size:
        peak = detach_start + int(np.argmax(detach_slice))
        after_peak = force_score[peak:release_index]
        below = np.flatnonzero(after_peak < max(0.25, force_score[peak] * 0.55))
        detach_index = peak + int(below[0]) if below.size else (peak + release_index) // 2
    else:
        detach_index = (lift_index + release_index) // 2
    detach_index = min(max(detach_index, lift_index + 1), release_index - 1)

    open_level = float(np.nanpercentile(bundle.gripper_position, 20))
    open_tolerance = max(0.05, float(np.nanpercentile(bundle.gripper_position, 80) - open_level) * 0.15)
    released_mask = (bundle.gripper_position <= open_level + open_tolerance) & (opening < 0.45)
    release_complete_index = _first_sustained(
        released_mask,
        release_index + 1,
        count_for(0.12),
    )
    if release_complete_index is None:
        release_complete_index = min(ts.size - 2, release_index + count_for(0.30))

    idle_mask = movement_after_grasp < 0.16
    retreat_index = None
    idle_count = count_for(0.35)
    for candidate in range(ts.size - idle_count, release_complete_index, -1):
        if np.all(idle_mask[candidate : candidate + idle_count]):
            retreat_index = candidate + idle_count - 1
            break
    retreat_index = retreat_index if retreat_index is not None else ts.size - 1

    indices = {
        "motion_start": motion_index,
        "contact_start": contact_index,
        "stable_grasp": stable_index,
        "lift_start": lift_index,
        "detach_complete": detach_index,
        "release_start": release_index,
        "release_complete": release_complete_index,
        "retreat_complete": retreat_index,
    }
    # Preserve a strict order even when a low-evidence fallback was necessary.
    previous = -1
    for name in EVENT_ORDER:
        indices[name] = min(ts.size - (len(EVENT_ORDER) - EVENT_ORDER.index(name)), max(indices[name], previous + 1))
        previous = indices[name]

    confidences = {
        "motion_start": float(np.clip(motion_score[motion_index], 0.10, 0.98)),
        "contact_start": float(np.clip(contact_fusion[contact_index], 0.08, 0.95)),
        "stable_grasp": float(
            np.clip(
                0.45 * gel_score[stable_index]
                + 0.35 * (1 - force_derivative[stable_index])
                + 0.2 * (1 - min(1.0, abs(bundle.gripper_velocity[stable_index]))),
                0.08,
                0.93,
            )
        ),
        "lift_start": float(np.clip(movement_after_grasp[lift_index], 0.08, 0.94)),
        "detach_complete": float(np.clip(0.35 + 0.4 * force_score[detach_index], 0.08, 0.82)),
        "release_start": float(np.clip(opening[release_index], 0.08, 0.97)),
        "release_complete": float(np.clip(0.45 + 0.25 * (1 - gel_score[release_complete_index]), 0.08, 0.88)),
        "retreat_complete": float(np.clip(0.65 * (1 - movement_after_grasp[retreat_index]), 0.08, 0.85)),
    }
    evidence = {
        "motion_start": {
            "joint_velocity_norm": float(bundle.joint_velocity_norm[motion_index]),
            "eef_linear_velocity": float(bundle.eef_linear_velocity[motion_index]),
            "fusion_score": float(motion_score[motion_index]),
        },
        "contact_start": {
            "closing_score": float(closing[contact_index]),
            "ati_force_score": float(force_score[contact_index]),
            "gelsight_contact_score": float(gel_score[contact_index]),
            "fusion_score": float(contact_fusion[contact_index]),
            "requires_multimodal_review": True,
        },
        "stable_grasp": {
            "gripper_velocity": float(bundle.gripper_velocity[stable_index]),
            "ati_derivative_score": float(force_derivative[stable_index]),
            "gelsight_contact_score": float(gel_score[stable_index]),
            "minimum_hold_seconds": config.stable_hold_min_seconds,
        },
        "lift_start": {
            "joint_or_eef_motion_score": float(movement_after_grasp[lift_index]),
            "eef_linear_velocity": float(bundle.eef_linear_velocity[lift_index]),
        },
        "detach_complete": {
            "ati_force_score": float(force_score[detach_index]),
            "method": "post-lift force peak/drop plus phase ordering",
            "manual_confirmation_required": True,
        },
        "release_start": {
            "opening_score": float(opening[release_index]),
            "gripper_target": float(bundle.gripper_target[release_index]),
        },
        "release_complete": {
            "gripper_position": float(bundle.gripper_position[release_complete_index]),
            "gelsight_contact_score": float(gel_score[release_complete_index]),
            "ati_force_score": float(force_score[release_complete_index]),
        },
        "retreat_complete": {
            "motion_score": float(movement_after_grasp[retreat_index]),
            "method": "final sustained idle",
        },
    }
    return [
        EventProposalValue(
            event_name=name,
            candidate_timestamp_ns=int(ts[indices[name]]),
            confidence=round(confidences[name], 4),
            evidence=evidence[name],
            detector_version=config.detector_version,
        )
        for name in EVENT_ORDER
    ]
