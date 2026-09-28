"""Read-only episode summaries for the collection UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    path: Path
    status: str
    duration_sec: float | None
    swatch_uid: str
    split: str
    camera_counts: tuple[int, int, int]
    ati_count: int
    ati_hz: float
    modified_ns: int


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _episode_status(path: Path, report: dict[str, Any]) -> str:
    if path.name.startswith(".") and path.name.endswith(".inprogress"):
        return "INCOMPLETE"
    if (path / "CAPTURE_INCOMPLETE.json").is_file():
        return "INCOMPLETE"
    if (path / "SENSOR_CAPTURE_COMPLETE.json").is_file() or (path / "COMPLETE.json").is_file():
        return "COMPLETE"
    if (path / "tactile/fabric_sidecar_COMPLETE.json").is_file():
        return "COMPLETE"
    if report.get("complete") is True:
        return "COMPLETE"
    if report:
        return "INCOMPLETE"
    return "UNKNOWN"


def _clock_count(report: dict[str, Any], name: str) -> int:
    clock = report.get("clock_reports", {}).get(name, {})
    return int(clock.get("count", 0)) if isinstance(clock, dict) else 0


def discover_episodes(output_root: Path) -> list[EpisodeSummary]:
    root = Path(output_root).expanduser()
    if not root.is_dir():
        return []
    summaries: list[EpisodeSummary] = []
    try:
        paths = list(root.iterdir())
    except OSError:
        return []
    for path in paths:
        try:
            if not path.is_dir():
                continue
            metadata_paths = sorted(path.glob("metadata_*.json"))
            report_path = path / "tactile/capture_report.json"
            if (
                not metadata_paths
                and not report_path.is_file()
                and "episode" not in path.name
            ):
                continue
            metadata = _read_json(metadata_paths[0]) if metadata_paths else {}
            report = _read_json(report_path)
            modified_ns = path.stat().st_mtime_ns
            status = _episode_status(path, report)
        except OSError:
            # Dataset roots can contain root-owned lost+found or mounted
            # directories. They are not episodes and must not crash the UI.
            continue
        duration = report.get("duration_sec")
        duration_sec = float(duration) if isinstance(duration, (float, int)) else None
        ati_clock = report.get("clock_reports", {}).get("ati_nano17", {})
        if not isinstance(ati_clock, dict):
            ati_clock = {}
        episode_id = str(metadata.get("episode_id") or path.name)
        if episode_id.startswith(".") and episode_id.endswith(".inprogress"):
            episode_id = episode_id[1 : -len(".inprogress")]
        summaries.append(
            EpisodeSummary(
                episode_id=episode_id,
                path=path.resolve(),
                status=status,
                duration_sec=duration_sec,
                swatch_uid=str(metadata.get("swatch_uid", "")),
                split=str(metadata.get("split", "")),
                camera_counts=(
                    _clock_count(report, "exterior_image_1_left"),
                    _clock_count(report, "wrist_image_left"),
                    _clock_count(report, "gelsight_left"),
                ),
                ati_count=int(ati_clock.get("count", 0)),
                ati_hz=float(ati_clock.get("measured_hz", 0.0)),
                modified_ns=modified_ns,
            )
        )
    return sorted(summaries, key=lambda value: (value.modified_ns, value.episode_id), reverse=True)
