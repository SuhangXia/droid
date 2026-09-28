"""Non-destructive timestamp repair and quality labels for Fabric-DROID episodes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from fabric_droid.io_utils import atomic_write_json
from fabric_droid.sync.clock import analyze_timestamps

QUALITY_SCHEMA_VERSION = "fabric-droid-quality-label-v1"
REPAIR_SCHEMA_VERSION = "fabric-droid-timeline-repair-v1"
D435_STREAMS = ("exterior_image_1_left", "wrist_image_left")


def _ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise RuntimeError(f"ffprobe found no video stream: {path}")
    stream = payload["streams"][0]

    def fraction(value: str | None) -> float:
        if not value:
            return 0.0
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)

    return {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "frame_count": int(stream.get("nb_frames", 0)),
        "duration_sec": float(payload.get("format", {}).get("duration", 0.0)),
        "size_bytes": int(payload.get("format", {}).get("size", path.stat().st_size)),
    }


def _timestamp_report(timestamps_ns: np.ndarray) -> dict[str, Any]:
    values = np.asarray(timestamps_ns, dtype=np.int64)
    clock = analyze_timestamps(values)
    gaps_ms = np.diff(values) / 1e6
    return {
        **clock.to_dict(),
        "gap_p50_ms": float(np.percentile(gaps_ms, 50)) if gaps_ms.size else 0.0,
        "gap_p95_ms": float(np.percentile(gaps_ms, 95)) if gaps_ms.size else 0.0,
        "gap_p99_ms": float(np.percentile(gaps_ms, 99)) if gaps_ms.size else 0.0,
        "gaps_over_80ms": int(np.count_nonzero(gaps_ms > 80.0)),
        "gaps_over_150ms": int(np.count_nonzero(gaps_ms > 150.0)),
    }


def _load_d435_timestamps(episode_dir: Path, name: str) -> np.ndarray:
    path = episode_dir / "recordings" / "timestamps" / f"{name}.npz"
    with np.load(path) as payload:
        return np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)


def _retime_stream_copy(
    source_video: Path,
    destination_video: Path,
    measured_fps: float,
) -> dict[str, Any]:
    before = _ffprobe(source_video)
    if measured_fps <= 0:
        raise RuntimeError(f"invalid measured fps for {source_video}: {measured_fps}")
    if abs(before["fps"] - measured_fps) <= 0.15:
        return {
            "method": "already_consistent_hardlink",
            "retimed": False,
            "before": before,
            "after": before,
        }
    scale = before["fps"] / measured_fps
    temporary = destination_video.with_name(f".{destination_video.stem}.retimed.mp4")
    if temporary.exists():
        raise FileExistsError(f"repair temporary already exists: {temporary}")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-itsscale",
            f"{scale:.12f}",
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    after = _ffprobe(temporary)
    if after["frame_count"] != before["frame_count"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"retime changed frame count for {source_video.name}: "
            f"{before['frame_count']} -> {after['frame_count']}"
        )
    if abs(after["fps"] - measured_fps) > 0.15:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"retime fps mismatch for {source_video.name}: "
            f"{after['fps']:.3f} vs {measured_fps:.3f}"
        )
    os.replace(temporary, destination_video)
    return {
        "method": "ffmpeg_itsscale_stream_copy_no_reencode",
        "retimed": True,
        "timestamp_scale": scale,
        "before": before,
        "after": after,
    }


def _event_report(episode_dir: Path) -> dict[str, Any]:
    path = episode_dir / "tactile" / "events.json"
    try:
        events = json.loads(path.read_text(encoding="utf-8")).get("events", [])
    except (OSError, json.JSONDecodeError):
        events = []
    names = sorted({str(event.get("name")) for event in events if event.get("name")})
    required = ("probe_complete", "release_time")
    return {
        "names": names,
        "probe_complete": "probe_complete" in names,
        "release_time": "release_time" in names,
        "segmentation_events_complete": all(name in names for name in required),
    }


def _quality_grade(streams: dict[str, dict[str, Any]]) -> str:
    rates = [float(stream["timestamps"]["measured_hz"]) for stream in streams.values()]
    max_gaps = [float(stream["timestamps"]["max_gap_ms"]) for stream in streams.values()]
    if min(rates) >= 28.0 and max(max_gaps) <= 80.0:
        return "A"
    if min(rates) >= 20.0 and max(max_gaps) <= 200.0:
        return "B_REPAIRED"
    if min(rates) >= 15.0 and max(max_gaps) <= 500.0:
        return "C_REVIEW"
    return "REJECT"


def repair_episode(source_episode: Path, destination_episode: Path) -> dict[str, Any]:
    """Create one hardlink-based repaired episode without changing the source."""

    source_episode = source_episode.resolve()
    destination_episode = destination_episode.resolve()
    if not (source_episode / "COMPLETE.json").is_file():
        raise ValueError(f"source episode is not complete: {source_episode}")
    if destination_episode.exists():
        raise FileExistsError(f"destination episode exists: {destination_episode}")
    staging = destination_episode.with_name(f".{destination_episode.name}.repairing")
    if staging.exists():
        raise FileExistsError(f"repair staging directory exists: {staging}")
    source_stats = {
        str(path.relative_to(source_episode)): {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in source_episode.rglob("*")
        if path.is_file()
    }
    shutil.copytree(source_episode, staging, copy_function=os.link)
    streams: dict[str, dict[str, Any]] = {}
    try:
        for name in D435_STREAMS:
            relative_video = Path("recordings") / "MP4" / f"{name}.mp4"
            source_video = source_episode / relative_video
            destination_video = staging / relative_video
            timestamps = _load_d435_timestamps(source_episode, name)
            timestamp_report = _timestamp_report(timestamps)
            repair = _retime_stream_copy(
                source_video,
                destination_video,
                float(timestamp_report["measured_hz"]),
            )
            if repair["after"]["frame_count"] != timestamp_report["count"]:
                raise RuntimeError(
                    f"{name} repaired video/timestamp count mismatch: "
                    f"{repair['after']['frame_count']} vs {timestamp_report['count']}"
                )
            streams[name] = {
                "timestamps": timestamp_report,
                "repair": repair,
            }

        duplicate = staging / "recordings" / "MP4" / "exterior_image_2_left.mp4"
        duplicate.unlink(missing_ok=True)
        os.link(
            staging / "recordings" / "MP4" / "exterior_image_1_left.mp4",
            duplicate,
        )

        events = _event_report(source_episode)
        trajectory_present = (source_episode / "trajectory.h5").is_file()
        grade = _quality_grade(streams)
        counts_match = all(
            stream["repair"]["after"]["frame_count"]
            == stream["timestamps"]["count"]
            for stream in streams.values()
        )
        timelines_match = all(
            abs(
                stream["repair"]["after"]["fps"]
                - stream["timestamps"]["measured_hz"]
            )
            <= 0.15
            for stream in streams.values()
        )
        full_trajectory_eligible = bool(
            trajectory_present
            and counts_match
            and timelines_match
            and grade != "REJECT"
        )
        segmented_eligible = bool(
            full_trajectory_eligible and events["segmentation_events_complete"]
        )
        recommendation = (
            "TRAIN_FULL_AND_SEGMENTED"
            if segmented_eligible
            else "TRAIN_FULL_TRAJECTORY_ONLY"
            if full_trajectory_eligible
            else "EXCLUDE"
        )
        label = {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "repair_schema_version": REPAIR_SCHEMA_VERSION,
            "episode_id": source_episode.name,
            "source_episode": str(source_episode),
            "repaired_episode": str(destination_episode),
            "source_immutable": True,
            "created_wall_time_ns": time.time_ns(),
            "quality_grade": grade,
            "recommendation": recommendation,
            "training_eligibility": {
                "full_trajectory": full_trajectory_eligible,
                "event_segmented": segmented_eligible,
            },
            "known_limitations": [
                "Missing source camera frames cannot be reconstructed.",
                "Repaired MP4 timing is constant-rate and derived from monotonic timestamp span.",
            ],
            "events": events,
            "d435_streams": streams,
            "checks": {
                "trajectory_present": trajectory_present,
                "video_timestamp_counts_match": counts_match,
                "encoded_fps_matches_timestamps": timelines_match,
                "source_files_checked_after_repair": False,
            },
        }
        atomic_write_json(staging / "DATA_QUALITY_LABEL.json", label)

        changed_sources = []
        for relative, before in source_stats.items():
            path = source_episode / relative
            after = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            if after != before:
                changed_sources.append(relative)
        if changed_sources:
            raise RuntimeError(
                f"source files changed during repair: {changed_sources[:5]}"
            )
        label["checks"]["source_files_checked_after_repair"] = True
        atomic_write_json(staging / "DATA_QUALITY_LABEL.json", label)
        os.replace(staging, destination_episode)
        return label
    except Exception as exc:
        atomic_write_json(
            staging / "REPAIR_FAILED.json",
            {"error": f"{type(exc).__name__}: {exc}", "source": str(source_episode)},
        )
        raise


def repair_dataset(source_root: Path, destination_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if source_root == destination_root:
        raise ValueError("repair destination must differ from source")
    destination_root.mkdir(parents=True, exist_ok=True)
    complete = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "COMPLETE.json").is_file()
    )
    incomplete = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "CAPTURE_INCOMPLETE.json").is_file()
    )
    labels: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for source_episode in complete:
        destination_episode = destination_root / source_episode.name
        if destination_episode.is_dir() and (
            destination_episode / "DATA_QUALITY_LABEL.json"
        ).is_file():
            existing_label = json.loads(
                (destination_episode / "DATA_QUALITY_LABEL.json").read_text(
                    encoding="utf-8"
                )
            )
            if existing_label.get("checks", {}).get(
                "encoded_fps_matches_timestamps",
                False,
            ):
                labels.append(existing_label)
                continue
            backup = destination_episode.with_name(
                f".{destination_episode.name}.pre-strict-fps-backup"
            )
            if backup.exists():
                failures.append(
                    {
                        "episode_id": source_episode.name,
                        "error": f"repair backup already exists: {backup}",
                    }
                )
                continue
            os.replace(destination_episode, backup)
            try:
                label = repair_episode(source_episode, destination_episode)
            except Exception as exc:
                if destination_episode.exists():
                    shutil.rmtree(destination_episode)
                os.replace(backup, destination_episode)
                failures.append(
                    {
                        "episode_id": source_episode.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                shutil.rmtree(backup)
                labels.append(label)
            continue
        try:
            labels.append(repair_episode(source_episode, destination_episode))
        except Exception as exc:
            failures.append(
                {
                    "episode_id": source_episode.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    manifest = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "created_wall_time_ns": time.time_ns(),
        "complete_source_count": len(complete),
        "repaired_count": len(labels),
        "failed_count": len(failures),
        "failures": failures,
        "quality_grade_counts": dict(
            Counter(label["quality_grade"] for label in labels)
        ),
        "recommendation_counts": dict(
            Counter(label["recommendation"] for label in labels)
        ),
        "incomplete_excluded": [
            {
                "episode_id": path.name,
                "label": "EXCLUDE_INCOMPLETE",
                "source_episode": str(path),
            }
            for path in incomplete
        ],
    }
    atomic_write_json(destination_root / "REPAIR_MANIFEST.json", manifest)
    return manifest
