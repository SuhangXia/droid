from __future__ import annotations

import json
from pathlib import Path

import cv2
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from fabric_droid_curator.backend.main import create_app
from fabric_droid_curator.config import CuratorConfig
from fabric_droid_curator.db.models import Episode, EventProposal, QCResult, SensorStream
from fabric_droid_curator.db.session import Database
from fabric_droid_curator.schemas import AnnotationPayload
from fabric_droid_curator.services.annotations import AnnotationService
from fabric_droid_curator.services.exports import ManifestExporter
from fabric_droid_curator.services.indexer import EpisodeIndexer
from fabric_droid_curator.services.lerobot_smoke import smoke_manifest

RAW_ROOT = Path("/home/suhang/datasets2/frabric_pi")
EPISODE_ID = "episode_20260727_163100"


@pytest.mark.skipif(not (RAW_ROOT / EPISODE_ID / "trajectory.h5").is_file(), reason="real episode unavailable")
def test_real_episode_read_only_end_to_end(tmp_path: Path) -> None:
    episode_dir = RAW_ROOT / EPISODE_ID
    observed = {
        path.relative_to(episode_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in episode_dir.rglob("*")
        if path.is_file()
    }
    cfg = CuratorConfig(
        data_root=RAW_ROOT,
        curation_root=tmp_path / "curation",
        reviewer="integration",
        splits={"train": ("swatch_001",), "validation": (), "heldout": ()},
    )
    database = Database(cfg.database_url)
    summary = EpisodeIndexer(cfg, database).scan(
        episode_ids={EPISODE_ID},
        generate_proxies=True,
        force=True,
    )
    assert summary == {
        "discovered": 1,
        "indexed": 1,
        "unchanged": 0,
        "skipped_incomplete": 0,
        "failed": 0,
        "failures": [],
    }
    with database.session() as db:
        episode = db.scalar(select(Episode).where(Episode.episode_id == EPISODE_ID))
        assert episode and episode.robot_frame_count == 511
        assert episode.d435_frame_count == episode.wrist_frame_count == 1094
        assert episode.gelsight_frame_count == 684
        assert episode.ati_sample_count == 18588
        streams = {item.name: item for item in db.scalars(select(SensorStream))}
        assert all(streams[name].monotonic for name in ("robot", "external", "wrist", "gelsight", "ati"))
        assert all(streams[name].decodable for name in ("external", "wrist", "gelsight"))
        proposals = list(db.scalars(select(EventProposal).order_by(EventProposal.candidate_timestamp_ns)))
        assert len(proposals) == 9
        assert {item.event_name for item in proposals}.issuperset({"motion_start", "trim_start"})
        assert all(item.evidence_json and item.detector_version for item in proposals)
        qc = db.scalar(select(QCResult))
        assert qc and json.loads(qc.hard_failures_json) == []

    for stream, expected in (("external", 1094), ("wrist", 1094), ("gelsight", 684)):
        path = cfg.curation_root / "proxies" / EPISODE_ID / f"{stream}.mp4"
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected
        ok, _ = capture.read()
        capture.release()
        assert ok

    client = TestClient(create_app(cfg))
    assert client.get("/api/health").json()["raw_data_read_only"] is True
    detail = client.get(f"/api/episodes/{EPISODE_ID}").json()
    assert detail["start_ns"] > detail["raw_start_ns"]
    assert detail["trimmed_leading_seconds"] > 0
    assert set(detail["streams"]).issuperset({"robot", "external", "wrist", "gelsight", "ati"})
    assert client.get(f"/api/episodes/{EPISODE_ID}/signals?max_points=200").status_code == 200
    media = client.get(f"/api/episodes/{EPISODE_ID}/media/external", headers={"Range": "bytes=0-1023"})
    assert media.status_code == 206 and len(media.content) == 1024

    service = AnnotationService(cfg, database)
    current = service.get_current(EPISODE_ID)
    payload = current["annotation"]
    payload.update(
        {
            "review_status": "verified",
            "dataset_decision": "keep",
            "action_branch": "remove",
            "single_layer": True,
            "stable_hold_present": True,
        }
    )
    saved = service.save(
        AnnotationPayload.model_validate(payload),
        reviewer="integration_fixture_not_human_ground_truth",
        reason="temporary real-data pipeline smoke",
        expected_version=0,
    )
    assert {item["segment_type"] for item in saved["segments"]} == {
        "grasp_probe",
        "remove_to_basket",
    }
    exported = ManifestExporter(cfg, database).export("fabric_droid_real_smoke_v001")
    manifest = Path(exported["output"]) / "manifest.parquet"
    assert pq.read_table(manifest).num_rows == 2
    smoke = smoke_manifest(manifest, data_root=RAW_ROOT, max_segments=2)
    assert len(smoke["segments"]) == 2
    assert all(item["finite"] and item["state_shape"][1] == 8 for item in smoke["segments"])
    assert all(item["pi05_batch"]["actions_shape"][1] == 8 for item in smoke["segments"])
    assert smoke["formal_training_started"] is False

    after = {
        path.relative_to(episode_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in episode_dir.rglob("*")
        if path.is_file()
    }
    assert after == observed
