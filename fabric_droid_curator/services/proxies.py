from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2

from fabric_droid_curator.config import ProxyConfig

from .readers import video_source


def ensure_video_proxy(
    episode_dir: Path,
    curation_root: Path,
    stream: str,
    config: ProxyConfig,
    *,
    force: bool = False,
) -> Path:
    output = curation_root / "proxies" / episode_dir.name / f"{stream}.mp4"
    source = video_source(episode_dir, stream)
    if output.is_file() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=output.parent, prefix=f".{stream}.", suffix=".mp4")
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"scale='min({config.width},iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        config.preset,
        "-crf",
        str(config.video_crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"proxy generation failed for {source}: {result.stderr.strip()}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def ensure_thumbnail(
    episode_dir: Path,
    curation_root: Path,
    seconds: float = 1.0,
    *,
    stream: str = "external",
    output_name: str | None = None,
    force: bool = False,
) -> Path:
    output = curation_root / "thumbnails" / (output_name or f"{episode_dir.name}.jpg")
    source = video_source(episode_dir, stream)
    if output.is_file() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = capture.read()
    if not ok:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode thumbnail source: {source}")
    temporary = output.with_name(f".{output.name}.tmp.jpg")
    try:
        if not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, 82]):
            raise RuntimeError(f"cannot write thumbnail: {temporary}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
