from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError
from sqlalchemy import inspect, select

from fabric_droid_curator.config import CuratorConfig, SegmentOverlapConfig
from fabric_droid_curator.constants import INSTRUCTION_TEMPLATES
from fabric_droid_curator.db.models import AuditLog, Episode, QCResult
from fabric_droid_curator.db.session import Database
from fabric_droid_curator.schemas import AnnotationPayload, EventValue
from fabric_droid_curator.services.annotations import AnnotationService
from fabric_droid_curator.services.events import detect_joint_angle_trim
from fabric_droid_curator.services.exports import ManifestExporter, leakage_audit
from fabric_droid_curator.services.indexer import legacy_action_branch, validate_split_config
from fabric_droid_curator.services.readers import camera_files_corrected, episode_is_complete, physical_camera_stream
from fabric_droid_curator.services.segments import build_segments, clamp_range


def config(tmp_path: Path, splits: dict[str, tuple[str, ...]] | None = None) -> CuratorConfig:
    return CuratorConfig(
        data_root=tmp_path / "raw",
        curation_root=tmp_path / "curation",
        splits=splits or {"train": ("swatch_001",), "validation": (), "heldout": ()},
    )


def annotation(notes: str = "", status: str = "unreviewed") -> AnnotationPayload:
    events = {
        name: EventValue(timestamp_ns=value, source="manual")
        for name, value in {
            "motion_start": 1_100_000_000,
            "contact_start": 1_300_000_000,
            "stable_grasp": 1_600_000_000,
            "branch_point": 2_000_000_000,
            "lift_start": 2_000_000_000,
            "detach_complete": 2_500_000_000,
            "release_start": 3_000_000_000,
            "release_complete": 3_500_000_000,
            "retreat_complete": 3_900_000_000,
        }.items()
    }
    return AnnotationPayload(
        episode_id="episode_test",
        session_id="session_test",
        swatch_uid="swatch_001",
        action_branch="remove",
        dataset_decision="keep",
        review_status=status,
        events=events,
        notes=notes,
    )


def seed_episode(database: Database) -> Episode:
    database.migrate()
    with database.session() as db:
        row = Episode(
            episode_id="episode_test",
            session_id="session_test",
            raw_path="/readonly/episode_test",
            duration_seconds=3.0,
            start_ns=1_000_000_000,
            end_ns=4_000_000_000,
            metadata_json=json.dumps({"success": True}),
            sensor_availability_json=json.dumps({"robot": True}),
            swatch_uid="swatch_001",
            split="train",
        )
        db.add(row)
        db.flush()
        episode_id = row.id
    with database.session() as db:
        return db.get(Episode, episode_id)


def test_annotation_schema_rejects_out_of_order_events() -> None:
    payload = annotation().model_dump(mode="json")
    payload["events"]["lift_start"]["timestamp_ns"] = payload["events"]["contact_start"]["timestamp_ns"]
    with pytest.raises(ValidationError, match="strict task order"):
        AnnotationPayload.model_validate(payload)


def test_overlap_clamp_and_prompt_mapping() -> None:
    result = build_segments(
        annotation(status="accepted"),
        annotation_version=4,
        episode_start_ns=1_000_000_000,
        episode_end_ns=4_000_000_000,
        overlap=SegmentOverlapConfig(pre_seconds=0.2, post_seconds=0.2, release_post_seconds=0.33),
    )
    values = {item.segment_type: item for item in result}
    assert set(values) == {"grasp_probe", "remove_to_basket"}
    assert values["grasp_probe"].start_ns == 1_000_000_000
    assert values["grasp_probe"].end_ns == 2_000_000_000
    assert values["remove_to_basket"].start_ns == 2_000_000_000
    assert values["remove_to_basket"].end_ns == 4_000_000_000
    assert values["remove_to_basket"].prompt == INSTRUCTION_TEMPLATES["remove_to_basket"]["prompt"]
    assert values["grasp_probe"].split_point_ns == 2_000_000_000
    assert values["grasp_probe"].tactile_window_start_ns == 1_300_000_000
    assert values["grasp_probe"].tactile_window_end_ns == 1_600_000_000
    assert clamp_range(0, 10, 2, 8) == (2, 8, ["segment_start_clamped_to_episode", "segment_end_clamped_to_episode"])


def test_drop_decision_and_incomplete_episode_filter() -> None:
    dropped = annotation(status="accepted").model_copy(update={"dataset_decision": "drop"})
    assert (
        build_segments(
            dropped,
            annotation_version=1,
            episode_start_ns=1_000_000_000,
            episode_end_ns=4_000_000_000,
            overlap=SegmentOverlapConfig(),
        )
        == []
    )
    assert episode_is_complete({"success": True, "failure_reason": ""})
    assert not episode_is_complete({"success": False, "failure_reason": "operator abort"})
    assert not episode_is_complete({"failure_reason": ""})


def test_joint_angle_trim_keeps_pre_roll() -> None:
    timestamps = np.arange(20, dtype=np.int64) * 100_000_000 + 1_000_000_000
    positions = np.zeros((20, 7), dtype=np.float64)
    positions[10:, 0] = np.linspace(0.003, 0.03, 10)
    proposal = detect_joint_angle_trim(timestamps, positions, pre_roll_seconds=0.2)
    assert proposal.event_name == "trim_start"
    assert proposal.candidate_timestamp_ns == timestamps[8]
    assert proposal.evidence["motion_onset_timestamp_ns"] == timestamps[10]


def test_camera_stream_name_correction_is_range_bounded() -> None:
    assert not camera_files_corrected("episode_20260727_224934")
    assert camera_files_corrected("episode_20260728_203423")
    assert camera_files_corrected("episode_20260728_233726")
    assert not camera_files_corrected("episode_20260728_233727")
    assert physical_camera_stream("episode_20260728_220000", "external") == "external"
    assert physical_camera_stream("episode_20260728_220000", "wrist") == "wrist"
    assert physical_camera_stream("episode_20260728_220000", "gelsight") == "gelsight"


def test_split_inheritance_and_leakage_detection() -> None:
    assert validate_split_config({"train": ("a",), "validation": ("b",), "heldout": ("c",)}) == {
        "a": "train",
        "b": "validation",
        "c": "heldout",
    }
    with pytest.raises(ValueError, match="leakage"):
        validate_split_config({"train": ("a",), "validation": ("a",)})
    rows = [
        Episode(episode_id="a1", raw_path="/a1", swatch_uid="a", split="train"),
        Episode(episode_id="a2", raw_path="/a2", swatch_uid="a", split="validation"),
    ]
    assert leakage_audit(rows, {})["pass"] is False


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"destination_tray": "green_left"}, "legacy_green_left"),
        ({"destination_tray": "white_right"}, "legacy_white_right"),
        ({"action_branch": "leave"}, "leave"),
        ({"destination_tray": "target_tray"}, "unknown"),
    ],
)
def test_legacy_action_mapping(metadata: dict[str, str], expected: str) -> None:
    assert legacy_action_branch(metadata) == expected


def test_sqlite_migration_annotation_versioning_and_restore(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    database = Database(cfg.database_url)
    seed_episode(database)
    service = AnnotationService(cfg, database)
    first = service.save(annotation("first"), reviewer="tester", expected_version=0)
    second = service.save(annotation("second"), reviewer="tester", expected_version=1)
    restored = service.restore("episode_test", 1, reviewer="tester", operation="undo")
    redone = service.restore("episode_test", 2, reviewer="tester", operation="redo")
    assert (first["version"], second["version"], restored["version"], redone["version"]) == (1, 2, 3, 4)
    assert restored["annotation"]["notes"] == "first"
    assert redone["annotation"]["notes"] == "second"
    assert (cfg.curation_root / "annotations" / "episode_test.v004.json").is_file()
    with database.session() as db:
        assert len(list(db.scalars(select(AuditLog)))) == 4
    tables = set(inspect(database.engine).get_table_names())
    assert {
        "episodes",
        "sensor_streams",
        "event_proposals",
        "annotations",
        "segments",
        "reviews",
        "exports",
        "camera_sessions",
        "qc_results",
        "schema_migrations",
    }.issubset(tables)


def test_immutable_manifest_export(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    database = Database(cfg.database_url)
    episode = seed_episode(database)
    with database.session() as db:
        db.add(
            QCResult(
                episode_pk=episode.id,
                qc_version="test",
                severity="green",
                scores_json=json.dumps({"overall_quality_score": 0.9}),
                hard_failures_json="[]",
                warnings_json="[]",
                metrics_json="{}",
            )
        )
    AnnotationService(cfg, database).save(
        annotation(status="verified"),
        reviewer="human",
        expected_version=0,
    )
    exported = ManifestExporter(cfg, database).export("fabric_droid_test_v001")
    output = Path(exported["output"])
    assert exported["segment_count"] == 2
    table = pq.read_table(output / "manifest.parquet")
    assert set(table["split"].to_pylist()) == {"train"}
    assert set(table["source_episode_id"].to_pylist()) == {"episode_test"}
    assert set(table["dataset_decision"].to_pylist()) == {"keep"}
    assert set(table["split_point_ns"].to_pylist()) == {2_000_000_000}
    assert (output / "leakage_audit.json").is_file()
    assert json.loads((output / "leakage_audit.json").read_text())["pass"]
    with pytest.raises(FileExistsError, match="immutable"):
        ManifestExporter(cfg, database).export("fabric_droid_test_v001")
