from __future__ import annotations

import json
from collections import Counter
from typing import Any

from sqlalchemy import select

from fabric_droid_curator.db.models import Episode, QCResult, Segment
from fabric_droid_curator.db.session import Database
from fabric_droid_curator.services.readers import episode_is_complete


def dashboard(database: Database) -> dict[str, Any]:
    with database.session() as db:
        episodes = [item for item in db.scalars(select(Episode)) if episode_is_complete(json.loads(item.metadata_json))]
        episode_ids = {item.id for item in episodes}
        qcs = [item for item in db.scalars(select(QCResult)) if item.episode_pk in episode_ids]
        segments = [
            item
            for item in db.scalars(select(Segment).where(Segment.active.is_(True)))
            if item.episode_pk in episode_ids
        ]
    qc_by_episode = {item.episode_pk: item for item in qcs}
    status = Counter(item.review_status for item in episodes)
    branches = Counter(item.action_branch for item in episodes)
    swatches = Counter(item.swatch_uid for item in episodes)
    cameras = Counter(item.camera_session_id for item in episodes)
    poses = Counter(item.start_pose_bucket for item in episodes)
    split_segments = Counter()
    episode_by_pk = {item.id: item for item in episodes}
    for segment in segments:
        split_segments[episode_by_pk[segment.episode_pk].split] += 1
    missing_sensor_count = 0
    holds: list[float] = []
    ati_peaks: list[float] = []
    severity = Counter()
    for episode in episodes:
        availability = json.loads(episode.sensor_availability_json)
        missing_sensor_count += int(not all(availability.values()))
        qc = qc_by_episode.get(episode.id)
        if qc:
            severity[qc.severity] += 1
            metrics = json.loads(qc.metrics_json)
            if metrics.get("hold_seconds") is not None:
                holds.append(float(metrics["hold_seconds"]))
            if metrics.get("ati_peak_force_delta") is not None:
                ati_peaks.append(float(metrics["ati_peak_force_delta"]))

    matrices: dict[str, list[dict[str, Any]]] = {}
    dimensions = {
        "swatch_action_branch": lambda item: item.action_branch,
        "swatch_camera_session": lambda item: item.camera_session_id,
        "swatch_start_pose": lambda item: item.start_pose_bucket,
    }
    for name, getter in dimensions.items():
        counts: Counter[tuple[str, str]] = Counter((item.swatch_uid, getter(item)) for item in episodes)
        matrices[name] = [
            {"swatch_uid": swatch, "column": column, "count": count}
            for (swatch, column), count in sorted(counts.items())
        ]
    segment_matrix: Counter[tuple[str, str]] = Counter()
    for segment in segments:
        segment_matrix[(episode_by_pk[segment.episode_pk].swatch_uid, segment.segment_type)] += 1
    matrices["swatch_segment_type"] = [
        {"swatch_uid": swatch, "column": column, "count": count}
        for (swatch, column), count in sorted(segment_matrix.items())
    ]
    return {
        "totals": {
            "episodes": len(episodes),
            "missing_sensor_episodes": missing_sensor_count,
            "segments": len(segments),
            "average_hold_seconds": round(sum(holds) / len(holds), 4) if holds else 0.0,
            **{f"qc_{name}": count for name, count in severity.items()},
        },
        "review_status": dict(status),
        "action_branches": dict(branches),
        "swatches": dict(swatches),
        "camera_sessions": dict(cameras),
        "start_pose_buckets": dict(poses),
        "splits": dict(split_segments),
        "distributions": {"hold_seconds": holds, "ati_peak_force": ati_peaks},
        "matrices": matrices,
    }
