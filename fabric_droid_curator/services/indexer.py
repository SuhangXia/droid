from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sqlalchemy import delete, select

from fabric_droid_curator.config import CuratorConfig
from fabric_droid_curator.db.models import (
    CameraSession,
    Episode,
    EventProposal,
    QCResult,
    SensorStream,
)
from fabric_droid_curator.db.session import Database

from .camera_sessions import compare_reference_frames
from .events import JOINT_TRIM_VERSION, EventProposalValue, detect_events, detect_joint_angle_trim
from .proxies import ensure_thumbnail, ensure_video_proxy
from .qc import QC_VERSION, compute_qc
from .readers import (
    ClockReport,
    episode_is_complete,
    inspect_h5,
    inspect_video,
    load_metadata,
    load_robot,
    load_stream_timestamps,
    raw_signature,
    video_source,
)
from .signals import get_or_compute_signals


def validate_split_config(splits: dict[str, tuple[str, ...]]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)
    aliases = {"heldout_test": "heldout", "val": "validation"}
    for raw_name, swatches in splits.items():
        name = aliases.get(raw_name, raw_name)
        if name not in {"train", "validation", "heldout"}:
            raise ValueError(f"unsupported split: {raw_name}")
        for swatch in swatches:
            if swatch in assignments and assignments[swatch] != name:
                duplicates[swatch].extend([assignments[swatch], name])
            assignments[swatch] = name
    if duplicates:
        detail = ", ".join(f"{swatch}={sorted(set(names))}" for swatch, names in duplicates.items())
        raise ValueError(f"swatch split leakage in config: {detail}")
    return assignments


def legacy_action_branch(metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("action_branch", "")).lower()
    if explicit in {"remove", "leave", "legacy_green_left", "legacy_white_right", "unknown"}:
        return explicit
    legacy = str(metadata.get("destination_tray", "")).lower()
    if legacy == "green_left":
        return "legacy_green_left"
    if legacy == "white_right":
        return "legacy_white_right"
    return "unknown"


def camera_session_id(metadata: dict[str, Any]) -> str:
    if metadata.get("camera_session_id"):
        return str(metadata["camera_session_id"])
    serials = metadata.get("camera_serials") or {}
    serial = str(serials.get("exterior_image_1_left", "unknown")).split(":")[0]
    return f"{metadata.get('session_id', 'unknown')}__{serial}"


def _stream_dict(
    *,
    name: str,
    kind: str,
    source_path: Path,
    timestamp_path: Path | None,
    timestamps: np.ndarray,
    video: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clock = ClockReport.from_timestamps(timestamps)
    video = video or {}
    return {
        "name": name,
        "kind": kind,
        "source_path": str(source_path),
        "timestamp_path": str(timestamp_path) if timestamp_path else None,
        "count": clock.count,
        "start_ns": clock.start_ns,
        "end_ns": clock.end_ns,
        "measured_hz": clock.measured_hz,
        "encoded_fps": video.get("fps"),
        "width": video.get("width"),
        "height": video.get("height"),
        "codec": video.get("codec"),
        "monotonic": clock.monotonic,
        "decodable": bool(video.get("decodable", True)),
        "max_gap_ms": clock.max_gap_ms,
        "details": {**(details or {}), "video": video, "clock": vars(clock)},
    }


def inspect_episode(episode_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metadata = load_metadata(episode_dir)
    h5 = inspect_h5(episode_dir)
    robot_ts = load_stream_timestamps(episode_dir, "robot")
    streams: dict[str, dict[str, Any]] = {
        "robot": _stream_dict(
            name="robot",
            kind="robot",
            source_path=episode_dir / "trajectory.h5",
            timestamp_path=None,
            timestamps=robot_ts,
            details={"h5": h5},
        )
    }
    streams["robot"]["finite"] = h5.get("finite", False)
    for name in ("external", "wrist", "gelsight"):
        source = video_source(episode_dir, name)
        timestamp_path = (
            episode_dir / "tactile" / "gelsight_left_timestamps.npy"
            if name == "gelsight"
            else episode_dir
            / "recordings"
            / "timestamps"
            / f"{'exterior_image_1_left' if name == 'external' else 'wrist_image_left'}.npz"
        )
        timestamps = load_stream_timestamps(episode_dir, name)
        video = inspect_video(source)
        streams[name] = _stream_dict(
            name=name,
            kind="video",
            source_path=source,
            timestamp_path=timestamp_path,
            timestamps=timestamps,
            video=video,
        )
        if video.get("frame_count") and video["frame_count"] != timestamps.size:
            streams[name]["details"]["frame_count_mismatch"] = True
    ati_ts = load_stream_timestamps(episode_dir, "ati")
    streams["ati"] = _stream_dict(
        name="ati",
        kind="force_torque",
        source_path=episode_dir / "tactile" / "ati_raw.parquet",
        timestamp_path=episode_dir / "tactile" / "ati_raw.parquet",
        timestamps=ati_ts,
    )
    start_ns = int(robot_ts[0]) if robot_ts.size else 0
    end_ns = int(robot_ts[-1]) if robot_ts.size else 0
    episode = {
        "episode_id": str(metadata.get("episode_id") or episode_dir.name),
        "session_id": str(metadata.get("session_id", "")),
        "raw_path": str(episode_dir.resolve()),
        "duration_seconds": max(0.0, (end_ns - start_ns) / 1e9),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "metadata": metadata,
    }
    return episode, streams


class EpisodeIndexer:
    def __init__(self, config: CuratorConfig, database: Database):
        self.config = config
        self.database = database
        self.split_assignment = validate_split_config(config.splits)
        self.config.ensure_derived_directories()
        self.database.migrate()

    def scan(
        self,
        *,
        episode_ids: set[str] | None = None,
        generate_proxies: bool | None = None,
        force: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        generate_proxies = self.config.proxy.enabled if generate_proxies is None else generate_proxies
        episode_dirs = sorted(
            path.parent
            for path in self.config.data_root.glob("episode_*/trajectory.h5")
            if not path.parent.name.startswith(".")
        )
        if episode_ids:
            episode_dirs = [path for path in episode_dirs if path.name in episode_ids]
        summary = {
            "discovered": len(episode_dirs),
            "indexed": 0,
            "unchanged": 0,
            "skipped_incomplete": 0,
            "failed": 0,
            "failures": [],
        }
        for number, episode_dir in enumerate(episode_dirs, 1):
            if progress:
                progress(f"[{number}/{len(episode_dirs)}] {episode_dir.name}")
            try:
                if not episode_is_complete(load_metadata(episode_dir)):
                    with self.database.session() as db:
                        existing = db.scalar(select(Episode).where(Episode.episode_id == episode_dir.name))
                        if existing is not None:
                            db.delete(existing)
                    summary["skipped_incomplete"] += 1
                    continue
                changed = self.index_one(
                    episode_dir,
                    generate_proxies=generate_proxies,
                    force=force,
                )
                summary["indexed" if changed else "unchanged"] += 1
            except Exception as exc:
                summary["failed"] += 1
                summary["failures"].append({"episode_id": episode_dir.name, "error": str(exc)})
        self.refresh_camera_sessions()
        return summary

    def index_one(self, episode_dir: Path, *, generate_proxies: bool, force: bool = False) -> bool:
        signature = raw_signature(episode_dir)
        robot = load_robot(episode_dir)
        trim_proposal = detect_joint_angle_trim(robot["timestamp_ns"], robot["joint_position"])
        with self.database.session() as db:
            existing = db.scalar(select(Episode).where(Episode.episode_id == episode_dir.name))
            unchanged = existing is not None and existing.raw_signature == signature and not force
        if unchanged:
            self._persist_trim_proposal(existing.id, trim_proposal)
            ensure_thumbnail(episode_dir, self.config.curation_root, self.config.proxy.thumbnail_time_seconds)
            if generate_proxies:
                for stream in ("external", "wrist", "gelsight"):
                    ensure_video_proxy(episode_dir, self.config.curation_root, stream, self.config.proxy)
            return False

        episode_data, stream_data = inspect_episode(episode_dir)
        metadata = episode_data["metadata"]
        swatch = str(metadata.get("swatch_uid", "unknown"))
        split = self.split_assignment.get(
            swatch, str(metadata.get("split", "unassigned")).replace("heldout_test", "heldout")
        )
        cam_session = camera_session_id(metadata)
        action_branch = legacy_action_branch(metadata)
        signal_path = self.config.curation_root / "signals" / f"{episode_dir.name}.npz"
        signals = get_or_compute_signals(episode_dir, signal_path, force=True)
        proposals = detect_events(signals, self.config.detector)
        proposals.append(trim_proposal)
        qc = compute_qc(
            episode=episode_data,
            streams=stream_data,
            signals=signals,
            proposals=proposals,
            config=self.config,
        )
        thumbnail = ensure_thumbnail(episode_dir, self.config.curation_root, self.config.proxy.thumbnail_time_seconds)
        if generate_proxies:
            for stream in ("external", "wrist", "gelsight"):
                ensure_video_proxy(episode_dir, self.config.curation_root, stream, self.config.proxy, force=force)

        with self.database.session() as db:
            row = db.scalar(select(Episode).where(Episode.episode_id == episode_data["episode_id"]))
            if row is None:
                row = Episode(episode_id=episode_data["episode_id"], raw_path=episode_data["raw_path"])
                db.add(row)
                db.flush()
            row.session_id = episode_data["session_id"]
            row.raw_path = episode_data["raw_path"]
            row.raw_signature = signature
            row.duration_seconds = episode_data["duration_seconds"]
            row.start_ns = episode_data["start_ns"]
            row.end_ns = episode_data["end_ns"]
            row.robot_frame_count = stream_data["robot"]["count"]
            row.d435_frame_count = stream_data["external"]["count"]
            row.wrist_frame_count = stream_data["wrist"]["count"]
            row.gelsight_frame_count = stream_data["gelsight"]["count"]
            row.ati_sample_count = stream_data["ati"]["count"]
            row.metadata_json = json.dumps(metadata, sort_keys=True)
            row.sensor_availability_json = json.dumps(
                {name: bool(item["count"]) for name, item in stream_data.items()},
                sort_keys=True,
            )
            row.camera_session_id = cam_session
            row.action_branch = row.action_branch if row.current_annotation_version else action_branch
            row.review_status = row.review_status if row.current_annotation_version else "auto_proposed"
            row.swatch_uid = swatch
            row.split = split
            row.start_pose_bucket = str(metadata.get("start_pose_bucket", "unknown"))
            row.indexed_at = datetime.now(timezone.utc)
            db.flush()

            db.execute(delete(SensorStream).where(SensorStream.episode_pk == row.id))
            for stream in stream_data.values():
                db.add(
                    SensorStream(
                        episode_pk=row.id,
                        name=stream["name"],
                        kind=stream["kind"],
                        source_path=stream["source_path"],
                        timestamp_path=stream["timestamp_path"],
                        count=stream["count"],
                        start_ns=stream["start_ns"],
                        end_ns=stream["end_ns"],
                        measured_hz=stream["measured_hz"],
                        encoded_fps=stream["encoded_fps"],
                        width=stream["width"],
                        height=stream["height"],
                        codec=stream["codec"],
                        monotonic=stream["monotonic"],
                        decodable=stream["decodable"],
                        details_json=json.dumps(stream["details"], sort_keys=True),
                    )
                )
            db.execute(
                delete(EventProposal).where(
                    EventProposal.episode_pk == row.id,
                    EventProposal.event_name == "trim_start",
                )
            )
            db.execute(
                delete(EventProposal).where(
                    EventProposal.episode_pk == row.id,
                    EventProposal.detector_version == self.config.detector.detector_version,
                )
            )
            for proposal in proposals:
                db.add(
                    EventProposal(
                        episode_pk=row.id,
                        event_name=proposal.event_name,
                        candidate_timestamp_ns=proposal.candidate_timestamp_ns,
                        confidence=proposal.confidence,
                        evidence_json=json.dumps(proposal.evidence, sort_keys=True),
                        detector_version=proposal.detector_version,
                    )
                )
            db.execute(delete(QCResult).where(QCResult.episode_pk == row.id, QCResult.qc_version == QC_VERSION))
            db.add(
                QCResult(
                    episode_pk=row.id,
                    qc_version=QC_VERSION,
                    severity=qc["severity"],
                    scores_json=json.dumps(qc["scores"], sort_keys=True),
                    hard_failures_json=json.dumps(qc["hard_failures"]),
                    warnings_json=json.dumps(qc["warnings"]),
                    metrics_json=json.dumps({**qc["metrics"], "thumbnail": str(thumbnail)}, sort_keys=True),
                )
            )
        return True

    def _persist_trim_proposal(self, episode_pk: int, proposal: EventProposalValue) -> None:
        with self.database.session() as db:
            current = db.scalar(
                select(EventProposal).where(
                    EventProposal.episode_pk == episode_pk,
                    EventProposal.event_name == "trim_start",
                    EventProposal.detector_version == JOINT_TRIM_VERSION,
                )
            )
            evidence_json = json.dumps(proposal.evidence, sort_keys=True)
            if (
                current is not None
                and current.candidate_timestamp_ns == proposal.candidate_timestamp_ns
                and current.confidence == proposal.confidence
                and current.evidence_json == evidence_json
            ):
                return
            db.execute(
                delete(EventProposal).where(
                    EventProposal.episode_pk == episode_pk,
                    EventProposal.event_name == "trim_start",
                )
            )
            db.add(
                EventProposal(
                    episode_pk=episode_pk,
                    event_name="trim_start",
                    candidate_timestamp_ns=proposal.candidate_timestamp_ns,
                    confidence=proposal.confidence,
                    evidence_json=evidence_json,
                    detector_version=proposal.detector_version,
                )
            )

    def refresh_camera_sessions(self) -> None:
        with self.database.session() as db:
            episodes = list(db.scalars(select(Episode).order_by(Episode.episode_id)))
            groups: dict[str, list[Episode]] = defaultdict(list)
            for episode in episodes:
                groups[episode.camera_session_id].append(episode)
            for session_id, members in groups.items():
                reference = members[0]
                reference_path = self.config.curation_root / "thumbnails" / f"{reference.episode_id}.jpg"
                comparisons = []
                for member in members[1:]:
                    candidate = self.config.curation_root / "thumbnails" / f"{member.episode_id}.jpg"
                    comparisons.append(compare_reference_frames(reference_path, candidate))
                valid = [item for item in comparisons if item["image_similarity"] is not None]

                def mean(key: str, rows: list[dict[str, Any]] = valid) -> float | None:
                    values = [item[key] for item in rows if item[key] is not None]
                    return float(np.mean(values)) if values else None

                classes = [item["consistency_class"] for item in comparisons]
                consistency = (
                    "major_shift"
                    if "major_shift" in classes
                    else ("minor_shift" if "minor_shift" in classes else "standard")
                )
                row = db.scalar(select(CameraSession).where(CameraSession.camera_session_id == session_id))
                if row is None:
                    row = CameraSession(
                        camera_session_id=session_id,
                        reference_episode_id=reference.episode_id,
                        reference_frame_path=str(reference_path),
                    )
                    db.add(row)
                row.reference_episode_id = reference.episode_id
                row.reference_frame_path = str(reference_path)
                row.episode_count = len(members)
                row.image_similarity = mean("image_similarity")
                row.estimated_translation = mean("estimated_translation")
                row.estimated_rotation = mean("estimated_rotation")
                row.homography_confidence = mean("homography_confidence")
                row.consistency_class = consistency
                row.details_json = json.dumps({"comparisons": comparisons}, sort_keys=True)
