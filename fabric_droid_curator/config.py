from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SegmentOverlapConfig:
    pre_seconds: float = 0.20
    post_seconds: float = 0.20
    release_post_seconds: float = 0.33


@dataclass(frozen=True)
class DetectorConfig:
    motion_min_seconds: float = 0.25
    contact_min_seconds: float = 0.12
    stable_hold_min_seconds: float = 0.25
    release_min_seconds: float = 0.12
    detector_version: str = "fabric-curator-fusion-v1"


@dataclass(frozen=True)
class ProxyConfig:
    enabled: bool = True
    width: int = 640
    video_crf: int = 28
    preset: str = "veryfast"
    thumbnail_time_seconds: float = 1.0


@dataclass(frozen=True)
class CuratorConfig:
    data_root: Path
    curation_root: Path
    reviewer: str = "local"
    segment_overlap: SegmentOverlapConfig = field(default_factory=SegmentOverlapConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    splits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    duration_min_seconds: float = 3.0
    duration_max_seconds: float = 180.0

    @property
    def database_path(self) -> Path:
        return self.curation_root / "curator.sqlite"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"

    def ensure_derived_directories(self) -> None:
        for name in ("annotations", "manifests", "exports", "proxies", "signals", "thumbnails", "logs"):
            (self.curation_root / name).mkdir(parents=True, exist_ok=True)


def _path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path | None = None, *, data_root: str | Path | None = None) -> CuratorConfig:
    config_path = Path(path or "configs/curator.yaml").expanduser().resolve()
    payload: dict[str, Any] = {}
    if config_path.is_file():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    configured_data_root = data_root or os.environ.get("FABRIC_DROID_CURATOR_DATA_ROOT") or payload.get("data_root")
    if not configured_data_root:
        raise ValueError("data_root is required in the config or on the command line")
    overlap = payload.get("segment_overlap", {})
    detector = payload.get("detector", {})
    proxy = payload.get("proxy", {})
    splits = {name: tuple(str(item) for item in values or ()) for name, values in (payload.get("splits") or {}).items()}
    return CuratorConfig(
        data_root=_path(configured_data_root, base),
        curation_root=_path(
            os.environ.get("FABRIC_DROID_CURATOR_ROOT") or payload.get("curation_root", "curation"), base
        ),
        reviewer=str(payload.get("reviewer", "local")),
        segment_overlap=SegmentOverlapConfig(**overlap),
        detector=DetectorConfig(**detector),
        proxy=ProxyConfig(**proxy),
        splits=splits,
        duration_min_seconds=float(payload.get("duration_min_seconds", 3.0)),
        duration_max_seconds=float(payload.get("duration_max_seconds", 180.0)),
    )
