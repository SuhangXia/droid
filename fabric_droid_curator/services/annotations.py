from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from fabric_droid_curator.config import CuratorConfig
from fabric_droid_curator.db.models import (
    Annotation,
    AuditLog,
    Episode,
    EventProposal,
    Review,
    Segment,
)
from fabric_droid_curator.db.session import Database
from fabric_droid_curator.schemas import AnnotationPayload, EventValue

from .segments import build_segments


class AnnotationConflictError(RuntimeError):
    pass


class AnnotationNotFoundError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AnnotationService:
    def __init__(self, config: CuratorConfig, database: Database):
        self.config = config
        self.database = database

    def default_annotation(self, episode_id: str) -> AnnotationPayload:
        with self.database.session() as db:
            episode = db.scalar(select(Episode).where(Episode.episode_id == episode_id))
            if episode is None:
                raise AnnotationNotFoundError(episode_id)
            metadata = json.loads(episode.metadata_json)
            proposals = list(
                db.scalars(
                    select(EventProposal)
                    .where(EventProposal.episode_pk == episode.id)
                    .order_by(EventProposal.candidate_timestamp_ns)
                )
            )
            events = {
                item.event_name: EventValue(
                    timestamp_ns=item.candidate_timestamp_ns,
                    source="automatic",
                    confidence=item.confidence,
                    evidence=json.loads(item.evidence_json),
                    detector_version=item.detector_version,
                )
                for item in proposals
            }
            if "trim_start" not in events:
                motion = events.get("motion_start")
                events["trim_start"] = EventValue(
                    timestamp_ns=max(episode.start_ns, motion.timestamp_ns - 200_000_000)
                    if motion
                    else episode.start_ns,
                    source="automatic",
                    confidence=motion.confidence if motion else 0.0,
                    evidence={"fallback": "motion_start_minus_pre_roll" if motion else "episode_start"},
                    detector_version=motion.detector_version if motion else None,
                )
            seed = events.get("stable_grasp")
            split_ns = seed.timestamp_ns if seed else episode.start_ns + (episode.end_ns - episode.start_ns) // 2
            events["branch_point"] = EventValue(
                timestamp_ns=split_ns,
                source="automatic",
                confidence=seed.confidence if seed else 0.0,
                evidence={"seed_event": "stable_grasp" if seed else "episode_midpoint"},
                detector_version=seed.detector_version if seed else None,
            )
            return AnnotationPayload(
                episode_id=episode.episode_id,
                session_id=episode.session_id,
                swatch_uid=episode.swatch_uid,
                source_slot=metadata.get("source_slot"),
                operator=metadata.get("operator"),
                camera_session_id=episode.camera_session_id,
                success=metadata.get("success"),
                review_status="unreviewed",
                failure_reason=str(metadata.get("failure_reason", "")),
                sensor_complete=all(json.loads(episode.sensor_availability_json).values()),
                action_branch=episode.action_branch,
                events=events,
            )

    def get_current(self, episode_id: str) -> dict[str, Any]:
        with self.database.session() as db:
            episode = db.scalar(select(Episode).where(Episode.episode_id == episode_id))
            if episode is None:
                raise AnnotationNotFoundError(episode_id)
            if not episode.current_annotation_version:
                return {"version": 0, "annotation": self.default_annotation(episode_id).model_dump(mode="json")}
            row = db.scalar(
                select(Annotation).where(
                    Annotation.episode_pk == episode.id,
                    Annotation.version == episode.current_annotation_version,
                )
            )
            if row is None:
                raise AnnotationNotFoundError(f"{episode_id} v{episode.current_annotation_version}")
            payload = json.loads(row.payload_json)
            if "trim_start" not in payload.get("events", {}):
                trim = db.scalar(
                    select(EventProposal)
                    .where(EventProposal.episode_pk == episode.id, EventProposal.event_name == "trim_start")
                    .order_by(EventProposal.created_at.desc())
                )
                motion = payload.get("events", {}).get("motion_start")
                timestamp_ns = (
                    trim.candidate_timestamp_ns
                    if trim
                    else max(episode.start_ns, motion["timestamp_ns"] - 200_000_000)
                    if motion
                    else episode.start_ns
                )
                payload.setdefault("events", {})["trim_start"] = {
                    "timestamp_ns": timestamp_ns,
                    "source": "automatic",
                    "confidence": trim.confidence if trim else motion.get("confidence", 0.0) if motion else 0.0,
                    "evidence": json.loads(trim.evidence_json)
                    if trim
                    else {"fallback": "motion_start_minus_pre_roll" if motion else "episode_start"},
                    "detector_version": trim.detector_version
                    if trim
                    else motion.get("detector_version")
                    if motion
                    else None,
                }
            if "branch_point" not in payload.get("events", {}):
                stable = payload.get("events", {}).get("stable_grasp")
                timestamp_ns = (
                    stable["timestamp_ns"] if stable else episode.start_ns + (episode.end_ns - episode.start_ns) // 2
                )
                payload.setdefault("events", {})["branch_point"] = {
                    "timestamp_ns": timestamp_ns,
                    "source": "automatic",
                    "confidence": stable.get("confidence", 0.0) if stable else 0.0,
                    "evidence": {"seed_event": "stable_grasp" if stable else "episode_midpoint"},
                    "detector_version": stable.get("detector_version") if stable else None,
                }
            return {
                "version": row.version,
                "annotation": payload,
                "reviewer": row.reviewer,
                "reason": row.reason,
                "created_at": row.created_at.isoformat(),
            }

    def history(self, episode_id: str) -> list[dict[str, Any]]:
        with self.database.session() as db:
            episode = db.scalar(select(Episode).where(Episode.episode_id == episode_id))
            if episode is None:
                raise AnnotationNotFoundError(episode_id)
            rows = list(
                db.scalars(
                    select(Annotation).where(Annotation.episode_pk == episode.id).order_by(Annotation.version.desc())
                )
            )
            return [
                {
                    "version": row.version,
                    "annotation": json.loads(row.payload_json),
                    "reviewer": row.reviewer,
                    "reason": row.reason,
                    "parent_version": row.parent_version,
                    "created_at": row.created_at.isoformat(),
                }
                for row in rows
            ]

    def save(
        self,
        annotation: AnnotationPayload,
        *,
        reviewer: str,
        reason: str = "",
        expected_version: int | None = None,
        parent_version: int | None = None,
    ) -> dict[str, Any]:
        payload = annotation.model_dump(mode="json")
        with self.database.session() as db:
            episode = db.scalar(select(Episode).where(Episode.episode_id == annotation.episode_id))
            if episode is None:
                raise AnnotationNotFoundError(annotation.episode_id)
            current = episode.current_annotation_version
            if expected_version is not None and expected_version != current:
                raise AnnotationConflictError(f"expected v{expected_version}, current is v{current}")
            previous = None
            if current:
                previous_row = db.scalar(
                    select(Annotation).where(
                        Annotation.episode_pk == episode.id,
                        Annotation.version == current,
                    )
                )
                previous = json.loads(previous_row.payload_json) if previous_row else None
            version = current + 1
            row = Annotation(
                episode_pk=episode.id,
                version=version,
                payload_json=json.dumps(payload, sort_keys=True),
                reviewer=reviewer,
                reason=reason,
                parent_version=current if parent_version is None else parent_version,
            )
            db.add(row)
            episode.current_annotation_version = version
            episode.review_status = annotation.review_status
            episode.action_branch = annotation.action_branch
            episode.swatch_uid = annotation.swatch_uid or episode.swatch_uid
            episode.camera_session_id = annotation.camera_session_id or episode.camera_session_id
            db.add(
                Review(
                    episode_pk=episode.id,
                    annotation_version=version,
                    status=annotation.review_status,
                    reviewer=reviewer,
                    reason=reason,
                )
            )
            db.add(
                AuditLog(
                    episode_pk=episode.id,
                    entity_type="annotation",
                    entity_id=annotation.episode_id,
                    action="create_version",
                    old_value_json=json.dumps(previous, sort_keys=True),
                    new_value_json=json.dumps(payload, sort_keys=True),
                    reviewer=reviewer,
                    reason=reason,
                    annotation_version=version,
                )
            )
            db.execute(delete(Segment).where(Segment.episode_pk == episode.id, Segment.active.is_(True)))
            segments = build_segments(
                annotation,
                annotation_version=version,
                episode_start_ns=episode.start_ns,
                episode_end_ns=episode.end_ns,
                overlap=self.config.segment_overlap,
            )
            for segment in segments:
                db.add(
                    Segment(
                        segment_id=segment.segment_id,
                        episode_pk=episode.id,
                        annotation_version=version,
                        segment_type=segment.segment_type,
                        start_ns=segment.start_ns,
                        end_ns=segment.end_ns,
                        prompt_template_id=segment.instruction_template_id,
                        prompt=segment.prompt,
                        payload_json=segment.model_dump_json(),
                    )
                )
        snapshot = {
            "schema_version": "fabric-droid-curator-annotation-v1",
            "annotation_version": version,
            "reviewer": reviewer,
            "reason": reason,
            "annotation": payload,
        }
        path = self.config.curation_root / "annotations" / f"{annotation.episode_id}.v{version:03d}.json"
        _atomic_json(path, snapshot)
        return {
            "version": version,
            "annotation": payload,
            "segments": [item.model_dump(mode="json") for item in segments],
        }

    def restore(
        self,
        episode_id: str,
        target_version: int,
        *,
        reviewer: str,
        operation: str,
    ) -> dict[str, Any]:
        if operation not in {"undo", "redo", "restore"}:
            raise ValueError("operation must be undo, redo, or restore")
        with self.database.session() as db:
            episode = db.scalar(select(Episode).where(Episode.episode_id == episode_id))
            if episode is None:
                raise AnnotationNotFoundError(episode_id)
            row = db.scalar(
                select(Annotation).where(
                    Annotation.episode_pk == episode.id,
                    Annotation.version == target_version,
                )
            )
            if row is None:
                raise AnnotationNotFoundError(f"{episode_id} v{target_version}")
            current = episode.current_annotation_version
            payload = AnnotationPayload.model_validate_json(row.payload_json)
        return self.save(
            payload,
            reviewer=reviewer,
            reason=f"__{operation}__:restore_v{target_version}",
            expected_version=current,
            parent_version=current,
        )
