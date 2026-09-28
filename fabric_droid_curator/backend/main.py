from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select

from fabric_droid_curator.config import CuratorConfig, load_config
from fabric_droid_curator.db.models import (
    CameraSession,
    Episode,
    EventProposal,
    QCResult,
    Segment,
    SensorStream,
)
from fabric_droid_curator.db.session import Database
from fabric_droid_curator.schemas import AnnotationPayload, SaveAnnotationRequest
from fabric_droid_curator.services.annotations import (
    AnnotationConflictError,
    AnnotationNotFoundError,
    AnnotationService,
)
from fabric_droid_curator.services.dashboard import dashboard
from fabric_droid_curator.services.proxies import ensure_thumbnail, ensure_video_proxy
from fabric_droid_curator.services.readers import (
    camera_files_corrected,
    episode_is_complete,
    load_stream_timestamps,
    physical_camera_stream,
)
from fabric_droid_curator.services.segments import build_segments
from fabric_droid_curator.services.signals import downsample_signals, get_or_compute_signals


def create_app(config: CuratorConfig | None = None) -> FastAPI:
    config = config or load_config(os.environ.get("FABRIC_DROID_CURATOR_CONFIG", "configs/curator.yaml"))
    config.ensure_derived_directories()
    database = Database(config.database_url)
    database.migrate()
    annotations = AnnotationService(config, database)
    app = FastAPI(title="Fabric-DROID Dataset Curator", version="0.1.0")
    app.state.config = config
    app.state.database = database
    app.state.annotations = annotations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with database.session() as db:
            count = len(list(db.scalars(select(Episode.id))))
        return {
            "status": "ok",
            "episode_count": count,
            "data_root": str(config.data_root),
            "curation_root": str(config.curation_root),
            "raw_data_read_only": True,
        }

    @app.get("/api/episodes")
    def list_episodes(
        review_status: str | None = None,
        action_branch: str | None = None,
        severity: str | None = None,
        search: str | None = None,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> dict[str, Any]:
        with database.session() as db:
            query = select(Episode).order_by(Episode.episode_id)
            if review_status:
                query = query.where(Episode.review_status == review_status)
            if action_branch:
                query = query.where(Episode.action_branch == action_branch)
            if search:
                query = query.where(Episode.episode_id.contains(search))
            rows = list(db.scalars(query))
            rows = [row for row in rows if episode_is_complete(json.loads(row.metadata_json))]
            qcs = {item.episode_pk: item for item in db.scalars(select(QCResult))}
            if severity:
                rows = [item for item in rows if qcs.get(item.id) and qcs[item.id].severity == severity]
            total = len(rows)
            rows = rows[offset : offset + limit]
            return {
                "total": total,
                "items": [
                    {
                        "episode_id": row.episode_id,
                        "session_id": row.session_id,
                        "duration_seconds": row.duration_seconds,
                        "review_status": row.review_status,
                        "action_branch": row.action_branch,
                        "swatch_uid": row.swatch_uid,
                        "split": row.split,
                        "camera_session_id": row.camera_session_id,
                        "start_pose_bucket": row.start_pose_bucket,
                        "annotation_version": row.current_annotation_version,
                        "severity": qcs[row.id].severity if row.id in qcs else "unknown",
                        "quality_score": (
                            json.loads(qcs[row.id].scores_json).get("overall_quality_score") if row.id in qcs else None
                        ),
                        "thumbnail_url": f"/api/episodes/{row.episode_id}/thumbnail",
                    }
                    for row in rows
                ],
            }

    def episode_trim_start(db, episode: Episode) -> int:  # type: ignore[no-untyped-def]
        proposal = db.scalar(
            select(EventProposal)
            .where(EventProposal.episode_pk == episode.id, EventProposal.event_name == "trim_start")
            .order_by(EventProposal.created_at.desc())
        )
        if proposal is None:
            return episode.start_ns
        return max(episode.start_ns, min(proposal.candidate_timestamp_ns, episode.end_ns - 1))

    def episode_or_404(db, episode_id: str) -> Episode:  # type: ignore[no-untyped-def]
        episode = db.scalar(select(Episode).where(Episode.episode_id == episode_id))
        if episode is None or not episode_is_complete(json.loads(episode.metadata_json)):
            raise HTTPException(404, f"episode not indexed: {episode_id}")
        return episode

    @app.get("/api/episodes/{episode_id}")
    def get_episode(episode_id: str) -> dict[str, Any]:
        with database.session() as db:
            episode = episode_or_404(db, episode_id)
            streams = list(db.scalars(select(SensorStream).where(SensorStream.episode_pk == episode.id)))
            proposals = list(
                db.scalars(
                    select(EventProposal)
                    .where(EventProposal.episode_pk == episode.id)
                    .order_by(EventProposal.candidate_timestamp_ns)
                )
            )
            qc = db.scalar(
                select(QCResult).where(QCResult.episode_pk == episode.id).order_by(QCResult.created_at.desc())
            )
            segments = list(
                db.scalars(
                    select(Segment)
                    .where(Segment.episode_pk == episode.id, Segment.active.is_(True))
                    .order_by(Segment.start_ns)
                )
            )
            trim_start_ns = episode_trim_start(db, episode)
            stream_by_name = {item.name: item for item in streams}
            result = {
                "episode_id": episode.episode_id,
                "session_id": episode.session_id,
                "raw_path": episode.raw_path,
                "duration_seconds": (episode.end_ns - trim_start_ns) / 1e9,
                "start_ns": trim_start_ns,
                "raw_start_ns": episode.start_ns,
                "trimmed_leading_seconds": (trim_start_ns - episode.start_ns) / 1e9,
                "camera_streams_swapped": camera_files_corrected(episode.episode_id),
                "end_ns": episode.end_ns,
                "review_status": episode.review_status,
                "action_branch": episode.action_branch,
                "swatch_uid": episode.swatch_uid,
                "split": episode.split,
                "camera_session_id": episode.camera_session_id,
                "metadata": json.loads(episode.metadata_json),
                "streams": {
                    logical_name: {
                        "kind": item.kind,
                        "recorded_stream_name": item.name,
                        "count": item.count,
                        "start_ns": item.start_ns,
                        "end_ns": item.end_ns,
                        "measured_hz": item.measured_hz,
                        "encoded_fps": item.encoded_fps,
                        "width": item.width,
                        "height": item.height,
                        "codec": item.codec,
                        "monotonic": item.monotonic,
                        "decodable": item.decodable,
                        "media_url": (
                            f"/api/episodes/{episode_id}/media/{logical_name}"
                            f"{'?camera_files=corrected-v1' if camera_files_corrected(episode_id) else ''}"
                            if item.kind == "video"
                            else None
                        ),
                    }
                    for logical_name in stream_by_name
                    for item in [stream_by_name[physical_camera_stream(episode.episode_id, logical_name)]]
                },
                "proposals": [
                    {
                        "event_name": item.event_name,
                        "candidate_timestamp_ns": item.candidate_timestamp_ns,
                        "confidence": item.confidence,
                        "evidence": json.loads(item.evidence_json),
                        "detector_version": item.detector_version,
                    }
                    for item in proposals
                ],
                "qc": (
                    {
                        "severity": qc.severity,
                        "scores": json.loads(qc.scores_json),
                        "hard_failures": json.loads(qc.hard_failures_json),
                        "warnings": json.loads(qc.warnings_json),
                        "metrics": json.loads(qc.metrics_json),
                    }
                    if qc
                    else None
                ),
                "segments": [json.loads(item.payload_json) for item in segments],
            }
        try:
            result["annotation"] = annotations.get_current(episode_id)
        except AnnotationNotFoundError:
            result["annotation"] = None
        return result

    @app.get("/api/episodes/{episode_id}/signals")
    def episode_signals(episode_id: str, max_points: int = Query(default=1600, ge=100, le=10000)) -> dict[str, Any]:
        with database.session() as db:
            episode = episode_or_404(db, episode_id)
            episode_dir = Path(episode.raw_path)
        path = config.curation_root / "signals" / f"{episode_id}.npz"
        bundle = get_or_compute_signals(episode_dir, path)
        return downsample_signals(bundle, max_points=max_points)

    @app.get("/api/episodes/{episode_id}/timestamps/{stream}")
    def episode_timestamps(episode_id: str, stream: str) -> dict[str, Any]:
        if stream not in {"external", "wrist", "gelsight", "robot", "ati"}:
            raise HTTPException(404, "unknown stream")
        with database.session() as db:
            episode = episode_or_404(db, episode_id)
            physical_stream = physical_camera_stream(episode_id, stream)
            stream_row = db.scalar(
                select(SensorStream).where(SensorStream.episode_pk == episode.id, SensorStream.name == physical_stream)
            )
            if stream_row is None:
                raise HTTPException(404, "stream not indexed")
            episode_dir = Path(episode.raw_path)
            encoded_fps = stream_row.encoded_fps or stream_row.measured_hz or 1.0
            trim_start_ns = episode_trim_start(db, episode)
        timestamps = load_stream_timestamps(episode_dir, physical_stream)
        return {
            "stream": stream,
            "timestamp_ns": timestamps.astype(np.int64).tolist(),
            "relative_seconds": ((timestamps - trim_start_ns) / 1e9).astype(float).tolist(),
            "media_seconds": (np.arange(timestamps.size) / encoded_fps).astype(float).tolist(),
            "encoded_fps": encoded_fps,
        }

    @app.get("/api/episodes/{episode_id}/media/{stream}")
    def episode_media(episode_id: str, stream: str) -> FileResponse:
        if stream not in {"external", "wrist", "gelsight"}:
            raise HTTPException(404, "unknown video stream")
        with database.session() as db:
            episode = episode_or_404(db, episode_id)
            episode_dir = Path(episode.raw_path)
        physical_stream = physical_camera_stream(episode_id, stream)
        proxy = config.curation_root / "proxies" / episode_id / f"{physical_stream}.mp4"
        if not proxy.is_file():
            try:
                proxy = ensure_video_proxy(episode_dir, config.curation_root, physical_stream, config.proxy)
            except Exception as exc:
                raise HTTPException(500, str(exc)) from exc
        return FileResponse(proxy, media_type="video/mp4", filename=f"{episode_id}_{stream}.mp4")

    @app.get("/api/episodes/{episode_id}/thumbnail")
    def episode_thumbnail(episode_id: str) -> FileResponse:
        with database.session() as db:
            episode = episode_or_404(db, episode_id)
            episode_dir = Path(episode.raw_path)
        physical_external = physical_camera_stream(episode_id, "external")
        output_name = f"{episode_id}.jpg" if physical_external == "external" else f"{episode_id}.logical_external.jpg"
        path = config.curation_root / "thumbnails" / output_name
        if not path.is_file():
            path = ensure_thumbnail(
                episode_dir,
                config.curation_root,
                config.proxy.thumbnail_time_seconds,
                stream=physical_external,
                output_name=output_name,
            )
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/episodes/{episode_id}/annotations/history")
    def annotation_history(episode_id: str) -> list[dict[str, Any]]:
        try:
            return annotations.history(episode_id)
        except AnnotationNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.put("/api/episodes/{episode_id}/annotation")
    def save_annotation(episode_id: str, request: SaveAnnotationRequest) -> dict[str, Any]:
        if request.annotation.episode_id != episode_id:
            raise HTTPException(400, "episode id does not match annotation payload")
        try:
            return annotations.save(
                request.annotation,
                reviewer=request.reviewer,
                reason=request.reason,
                expected_version=request.expected_version,
            )
        except AnnotationConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except AnnotationNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/episodes/{episode_id}/annotations/{operation}/{target_version}")
    def restore_annotation(
        episode_id: str, operation: str, target_version: int, reviewer: str = "local"
    ) -> dict[str, Any]:
        try:
            return annotations.restore(
                episode_id,
                target_version,
                reviewer=reviewer,
                operation=operation,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except AnnotationNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/episodes/{episode_id}/segments/preview")
    def preview_segments(episode_id: str, annotation: AnnotationPayload) -> list[dict[str, Any]]:
        with database.session() as db:
            episode = episode_or_404(db, episode_id)
            version = episode.current_annotation_version + 1
            values = build_segments(
                annotation,
                annotation_version=version,
                episode_start_ns=episode.start_ns,
                episode_end_ns=episode.end_ns,
                overlap=config.segment_overlap,
                allow_draft=True,
            )
        return [item.model_dump(mode="json") for item in values]

    @app.get("/api/dashboard")
    def get_dashboard() -> dict[str, Any]:
        return dashboard(database)

    @app.get("/api/camera-sessions")
    def camera_sessions() -> list[dict[str, Any]]:
        with database.session() as db:
            rows = list(db.scalars(select(CameraSession).order_by(CameraSession.camera_session_id)))
            return [
                {
                    "camera_session_id": row.camera_session_id,
                    "reference_episode_id": row.reference_episode_id,
                    "reference_frame_url": f"/api/episodes/{row.reference_episode_id}/thumbnail",
                    "episode_count": row.episode_count,
                    "image_similarity": row.image_similarity,
                    "estimated_translation": row.estimated_translation,
                    "estimated_rotation": row.estimated_rotation,
                    "homography_confidence": row.homography_confidence,
                    "consistency_class": row.consistency_class,
                }
                for row in rows
            ]

    return app


app = create_app()
