from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sqlalchemy import select

from fabric_droid_curator.config import CuratorConfig
from fabric_droid_curator.constants import INSTRUCTION_TEMPLATES
from fabric_droid_curator.db.models import (
    Annotation,
    CameraSession,
    Episode,
    Export,
    QCResult,
    Segment,
)
from fabric_droid_curator.db.session import Database
from fabric_droid_curator.services.readers import camera_files_corrected, episode_is_complete

MANIFEST_SCHEMA = pa.schema(
    [
        ("segment_id", pa.string()),
        ("source_episode_id", pa.string()),
        ("segment_type", pa.string()),
        ("start_ns", pa.int64()),
        ("end_ns", pa.int64()),
        ("duration", pa.float64()),
        ("instruction_template_id", pa.string()),
        ("prompt", pa.string()),
        ("dataset_decision", pa.string()),
        ("action_branch", pa.string()),
        ("swatch_uid", pa.string()),
        ("split", pa.string()),
        ("camera_session_id", pa.string()),
        ("camera_streams_swapped", pa.bool_()),
        ("quality_score", pa.float64()),
        ("annotation_version", pa.int64()),
        ("export_version", pa.string()),
        ("split_point_ns", pa.int64()),
        ("contact_start_ns", pa.int64()),
        ("stable_grasp_ns", pa.int64()),
        ("tactile_window_start_ns", pa.int64()),
        ("tactile_window_end_ns", pa.int64()),
    ]
)


def leakage_audit(episodes: list[Episode], configured_splits: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    observed: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        observed[episode.swatch_uid].add(episode.split)
    for split, swatches in configured_splits.items():
        normalized = "heldout" if split == "heldout_test" else split
        for swatch in swatches:
            observed[swatch].add(normalized)
    leaked = {swatch: sorted(splits) for swatch, splits in observed.items() if len(splits) > 1}
    return {
        "pass": not leaked,
        "unit": "swatch_uid",
        "swatch_assignments": {key: sorted(value) for key, value in sorted(observed.items())},
        "leaked_swatches": leaked,
        "heldout_excluded_from_training_export": True,
        "heldout_used_for_normalization": False,
    }


class ManifestExporter:
    def __init__(self, config: CuratorConfig, database: Database):
        self.config = config
        self.database = database

    def export(self, export_version: str, *, include_heldout: bool = False) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", export_version):
            raise ValueError("invalid export version")
        output = self.config.curation_root / "exports" / export_version
        if output.exists():
            raise FileExistsError(f"immutable export already exists: {output}")
        with self.database.session() as db:
            if db.scalar(select(Export).where(Export.export_version == export_version)):
                raise FileExistsError(f"export version already recorded: {export_version}")
            episodes = [
                item for item in db.scalars(select(Episode)) if episode_is_complete(json.loads(item.metadata_json))
            ]
            episode_by_pk = {item.id: item for item in episodes}
            segments = [
                item
                for item in db.scalars(select(Segment).where(Segment.active.is_(True)))
                if item.episode_pk in episode_by_pk
            ]
            annotations = list(db.scalars(select(Annotation)))
            qcs = list(db.scalars(select(QCResult)))
            camera_sessions = list(db.scalars(select(CameraSession)))
        audit = leakage_audit(episodes, self.config.splits)
        if not audit["pass"]:
            raise ValueError(f"export blocked by swatch leakage: {audit['leaked_swatches']}")
        qc_by_episode = {item.episode_pk: item for item in qcs}
        rows: list[dict[str, Any]] = []
        excluded_heldout = 0
        for segment in segments:
            episode = episode_by_pk[segment.episode_pk]
            if episode.split == "heldout" and not include_heldout:
                excluded_heldout += 1
                continue
            payload = json.loads(segment.payload_json)
            score = (
                json.loads(qc_by_episode[episode.id].scores_json).get("overall_quality_score", 0.0)
                if episode.id in qc_by_episode
                else 0.0
            )
            rows.append(
                {
                    "segment_id": segment.segment_id,
                    "source_episode_id": episode.episode_id,
                    "segment_type": segment.segment_type,
                    "start_ns": segment.start_ns,
                    "end_ns": segment.end_ns,
                    "duration": (segment.end_ns - segment.start_ns) / 1e9,
                    "instruction_template_id": segment.prompt_template_id,
                    "prompt": segment.prompt,
                    "dataset_decision": payload.get("dataset_decision", "undecided"),
                    "action_branch": episode.action_branch,
                    "swatch_uid": episode.swatch_uid,
                    "split": episode.split,
                    "camera_session_id": episode.camera_session_id,
                    "camera_streams_swapped": camera_files_corrected(episode.episode_id),
                    "quality_score": float(score),
                    "annotation_version": segment.annotation_version,
                    "export_version": export_version,
                    "split_point_ns": payload.get("split_point_ns"),
                    "contact_start_ns": payload.get("contact_start_ns"),
                    "stable_grasp_ns": payload.get("stable_grasp_ns"),
                    "tactile_window_start_ns": payload.get("tactile_window_start_ns"),
                    "tactile_window_end_ns": payload.get("tactile_window_end_ns"),
                }
            )
        annotation_by_episode_version = {(item.episode_pk, item.version): item for item in annotations}
        snapshot = []
        for episode in episodes:
            item = annotation_by_episode_version.get((episode.id, episode.current_annotation_version))
            if item:
                snapshot.append(
                    {
                        "source_episode_id": episode.episode_id,
                        "raw_path": episode.raw_path,
                        "annotation_version": item.version,
                        "reviewer": item.reviewer,
                        "created_at": item.created_at.isoformat(),
                        "annotation": json.loads(item.payload_json),
                    }
                )
        rejected = [
            {"episode_id": item.episode_id, "annotation_version": item.current_annotation_version}
            for item in episodes
            if item.review_status == "rejected"
        ]
        recovery = [
            {"episode_id": item.episode_id, "annotation_version": item.current_annotation_version}
            for item in episodes
            if item.review_status == "recovery"
        ]
        stats = {
            "segment_count": len(rows),
            "duration_seconds": round(sum(item["duration"] for item in rows), 6),
            "by_segment_type": dict(Counter(item["segment_type"] for item in rows)),
            "by_split": dict(Counter(item["split"] for item in rows)),
            "by_action_branch": dict(Counter(item["action_branch"] for item in rows)),
            "excluded_heldout_segments": excluded_heldout,
        }
        export_config = {
            "export_version": export_version,
            "data_root": str(self.config.data_root),
            "curation_root": str(self.config.curation_root),
            "include_heldout": include_heldout,
            "segment_overlap": vars(self.config.segment_overlap),
            "splits": {key: list(value) for key, value in self.config.splits.items()},
            "normalization_fitted": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{export_version}.", dir=output.parent))
        try:
            table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
            pq.write_table(table, staging / "manifest.parquet", compression="zstd")
            _json(staging / "annotation_snapshot.json", snapshot)
            _json(staging / "instruction_templates.json", INSTRUCTION_TEMPLATES)
            _json(
                staging / "split_manifest.json",
                {
                    "splits": {key: list(value) for key, value in self.config.splits.items()},
                    "segment_assignments": {item["segment_id"]: item["split"] for item in rows},
                },
            )
            _json(staging / "segment_stats.json", stats)
            _json(staging / "rejected_episodes.json", rejected)
            _json(staging / "recovery_episodes.json", recovery)
            _json(staging / "leakage_audit.json", audit)
            _json(
                staging / "sensor_qc_report.json",
                [
                    {
                        "episode_id": episode_by_pk[item.episode_pk].episode_id,
                        "severity": item.severity,
                        "scores": json.loads(item.scores_json),
                        "hard_failures": json.loads(item.hard_failures_json),
                        "warnings": json.loads(item.warnings_json),
                        "metrics": json.loads(item.metrics_json),
                    }
                    for item in qcs
                    if item.episode_pk in episode_by_pk
                ],
            )
            _json(
                staging / "camera_session_report.json",
                [
                    {
                        "camera_session_id": item.camera_session_id,
                        "reference_episode_id": item.reference_episode_id,
                        "reference_frame_path": item.reference_frame_path,
                        "episode_count": item.episode_count,
                        "image_similarity": item.image_similarity,
                        "estimated_translation": item.estimated_translation,
                        "estimated_rotation": item.estimated_rotation,
                        "homography_confidence": item.homography_confidence,
                        "consistency_class": item.consistency_class,
                    }
                    for item in camera_sessions
                ],
            )
            (staging / "export_config.yaml").write_text(
                yaml.safe_dump(export_config, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(staging, output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        with self.database.session() as db:
            db.add(
                Export(
                    export_version=export_version,
                    output_path=str(output),
                    config_json=json.dumps(export_config, sort_keys=True),
                    segment_count=len(rows),
                    leakage_audit_json=json.dumps(audit, sort_keys=True),
                )
            )
        return {"output": str(output), **stats, "leakage_audit": audit}


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
