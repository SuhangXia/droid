from __future__ import annotations

from typing import Any

import numpy as np

from fabric_droid_curator.config import CuratorConfig

from .events import EventProposalValue
from .signals import SignalBundle

QC_VERSION = "fabric-curator-qc-v1"


def _clip_score(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def compute_qc(
    *,
    episode: dict[str, Any],
    streams: dict[str, dict[str, Any]],
    signals: SignalBundle,
    proposals: list[EventProposalValue],
    config: CuratorConfig,
) -> dict[str, Any]:
    hard: list[str] = []
    warnings: list[str] = []
    required = ("robot", "external", "wrist", "gelsight", "ati")
    for name in required:
        stream = streams.get(name)
        if not stream or stream.get("count", 0) == 0:
            hard.append(f"missing_or_empty_stream:{name}")
        if name in {"external", "wrist", "gelsight"} and stream and not stream.get("decodable", False):
            hard.append(f"video_not_decodable:{name}")
        if stream and stream.get("count", 0) and not stream.get("monotonic", False):
            hard.append(f"timestamp_not_monotonic:{name}")
    robot = streams.get("robot", {})
    if not robot.get("finite", True):
        hard.append("robot_state_or_action_nan")
    duration = float(episode.get("duration_seconds", 0.0))
    if not config.duration_min_seconds <= duration <= config.duration_max_seconds:
        hard.append("episode_duration_out_of_range")

    gripper_range = float(np.ptp(signals.gripper_position)) if signals.gripper_position.size else 0.0
    if gripper_range < 0.15:
        hard.append("no_gripper_closing")
    contact = next((item for item in proposals if item.event_name == "contact_start"), None)
    if contact is None or contact.confidence < 0.12:
        hard.append("no_multimodal_contact_evidence")

    stream_count = sum(
        bool(streams.get(name, {}).get("count", 0)) and bool(streams.get(name, {}).get("monotonic", False))
        for name in required
    )
    sensor_completeness = stream_count / len(required)
    timestamp_sync = 1.0
    gaps: list[float] = []
    for name in ("external", "wrist", "gelsight"):
        stream = streams.get(name, {})
        expected = float(stream.get("measured_hz", 0.0))
        encoded = float(stream.get("encoded_fps") or 0.0)
        if expected and encoded and abs(expected - encoded) > 0.5:
            warnings.append(f"encoded_fps_differs_from_timestamp_rate:{name}")
        max_gap = float(stream.get("max_gap_ms", 0.0))
        if max_gap > 100.0:
            warnings.append(f"large_frame_gap:{name}:{max_gap:.1f}ms")
        gaps.append(max_gap)
    timestamp_sync = _clip_score(1.0 - max(gaps, default=0.0) / 500.0)

    event_map = {item.event_name: item for item in proposals}
    stable = event_map.get("stable_grasp")
    lift = event_map.get("lift_start")
    release = event_map.get("release_complete")
    hold_seconds = (lift.candidate_timestamp_ns - stable.candidate_timestamp_ns) / 1e9 if stable and lift else 0.0
    if hold_seconds < 0.25:
        warnings.append("stable_hold_shorter_than_0.25s")
    contact_to_lift = (lift.candidate_timestamp_ns - contact.candidate_timestamp_ns) / 1e9 if contact and lift else 0.0
    if 0 < contact_to_lift < 0.25:
        warnings.append("contact_to_lift_interval_too_short")
    if signals.gripper_velocity.size and np.sum(np.abs(signals.gripper_velocity) > 1.0) > 6:
        warnings.append("gripper_jitter_or_regrasp")
    if duration and signals.joint_velocity_norm.size:
        idle_fraction = float(np.mean(signals.joint_velocity_norm < 0.015))
        if idle_fraction > 0.55:
            warnings.append("long_idle")
    else:
        idle_fraction = 1.0

    contact_quality = contact.confidence if contact else 0.0
    stable_hold_score = _clip_score(hold_seconds / 0.5) * (stable.confidence if stable else 0.0)
    motion_smoothness = (
        _clip_score(1.0 - float(np.nanpercentile(np.abs(np.diff(signals.joint_velocity_norm)), 95)) / 0.25)
        if signals.joint_velocity_norm.size > 2
        else 0.0
    )
    grasp_stability = _clip_score((stable.confidence if stable else 0.0) * (1.0 - min(1.0, idle_fraction * 0.2)))
    camera_consistency = timestamp_sync
    task_success = float(bool(episode.get("metadata", {}).get("success", False) and release))
    scores = {
        "sensor_completeness_score": _clip_score(sensor_completeness),
        "timestamp_sync_score": timestamp_sync,
        "contact_quality_score": _clip_score(contact_quality),
        "stable_hold_score": _clip_score(stable_hold_score),
        "motion_smoothness_score": motion_smoothness,
        "grasp_stability_score": grasp_stability,
        "camera_consistency_score": camera_consistency,
        "task_success_score": task_success,
    }
    scores["overall_quality_score"] = _clip_score(sum(scores.values()) / len(scores) * (0.35 if hard else 1.0))
    severity = "red" if hard else ("yellow" if warnings or scores["overall_quality_score"] < 0.72 else "green")
    return {
        "qc_version": QC_VERSION,
        "severity": severity,
        "scores": scores,
        "hard_failures": hard,
        "warnings": warnings,
        "metrics": {
            "hold_seconds": round(hold_seconds, 4),
            "contact_to_lift_seconds": round(contact_to_lift, 4),
            "idle_fraction": round(idle_fraction, 4),
            "gripper_range": round(gripper_range, 4),
            "ati_peak_force_delta": round(float(np.max(signals.ati_force_delta)), 5)
            if signals.ati_force_delta.size
            else None,
            "release_detected": release is not None,
        },
    }
