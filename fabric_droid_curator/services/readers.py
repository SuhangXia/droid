from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.parquet as pq

from fabric_droid_curator.constants import TIMESTAMP_STREAM_PATHS, VIDEO_STREAM_PATHS

CAMERA_SWAP_FIRST_EPISODE = "episode_20260728_203423"
CAMERA_SWAP_LAST_EPISODE = "episode_20260728_233726"


@dataclass
class ClockReport:
    count: int
    start_ns: int
    end_ns: int
    measured_hz: float
    monotonic: bool
    duplicate_count: int
    max_gap_ms: float

    @classmethod
    def from_timestamps(cls, values: np.ndarray) -> "ClockReport":
        timestamps = np.asarray(values, dtype=np.int64).reshape(-1)
        if not timestamps.size:
            return cls(0, 0, 0, 0.0, False, 0, 0.0)
        gaps = np.diff(timestamps)
        duration_ns = int(timestamps[-1] - timestamps[0])
        return cls(
            count=int(timestamps.size),
            start_ns=int(timestamps[0]),
            end_ns=int(timestamps[-1]),
            measured_hz=float((timestamps.size - 1) * 1e9 / duration_ns) if duration_ns > 0 else 0.0,
            monotonic=bool(timestamps.size == 1 or np.all(gaps > 0)),
            duplicate_count=int(np.sum(gaps == 0)),
            max_gap_ms=float(gaps.max() / 1e6) if gaps.size else 0.0,
        )


def load_metadata(episode_dir: Path) -> dict[str, Any]:
    paths = sorted(episode_dir.glob("metadata_*.json"))
    if len(paths) != 1:
        return {}
    return json.loads(paths[0].read_text(encoding="utf-8"))


def episode_is_complete(metadata: dict[str, Any]) -> bool:
    """Return true only for explicitly successful, non-failed recordings."""
    return metadata.get("success") is True and not str(metadata.get("failure_reason", "")).strip()


def camera_files_corrected(episode_id: str) -> bool:
    return CAMERA_SWAP_FIRST_EPISODE <= episode_id <= CAMERA_SWAP_LAST_EPISODE


def physical_camera_stream(_episode_id: str, logical_stream: str) -> str:
    """Camera files are physically corrected, so logical and recorded names match."""
    return logical_stream


def load_stream_timestamps(episode_dir: Path, stream: str) -> np.ndarray:
    if stream == "robot":
        with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
            return np.asarray(handle["observation/timestamp/monotonic_ns"], dtype=np.int64)
    if stream == "ati":
        path = episode_dir / "tactile" / "ati_raw.parquet"
        if not path.is_file():
            return np.asarray([], dtype=np.int64)
        table = pq.read_table(path, columns=["timestamp_monotonic_ns"])
        return table["timestamp_monotonic_ns"].to_numpy(zero_copy_only=False).astype(np.int64)
    path = episode_dir / TIMESTAMP_STREAM_PATHS[stream]
    if not path.is_file():
        return np.asarray([], dtype=np.int64)
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.int64)
    with np.load(path) as payload:
        return np.asarray(payload["timestamp_monotonic_ns"], dtype=np.int64)


def load_robot(episode_dir: Path) -> dict[str, np.ndarray]:
    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:

        def take(name: str) -> np.ndarray:
            return np.asarray(handle[name])

        return {
            "timestamp_ns": take("observation/timestamp/monotonic_ns").astype(np.int64),
            "joint_position": take("observation/robot_state/joint_positions").astype(np.float64),
            "joint_velocity": take("observation/robot_state/joint_velocities").astype(np.float64),
            "eef_pose": take("observation/robot_state/cartesian_position").astype(np.float64),
            "gripper_position": take("observation/robot_state/gripper_position").astype(np.float64),
            "joint_velocity_action": take("action/joint_velocity").astype(np.float64),
            "gripper_target": take("action/gripper_position").astype(np.float64),
        }


def load_ati(episode_dir: Path) -> dict[str, np.ndarray]:
    path = episode_dir / "tactile" / "ati_raw.parquet"
    names = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    empty = {
        "timestamp_ns": np.asarray([], dtype=np.int64),
        **{name: np.asarray([], dtype=np.float64) for name in names},
    }
    if not path.is_file():
        return empty
    table = pq.read_table(path)
    if not table.num_rows:
        return empty
    return {
        "timestamp_ns": table["timestamp_monotonic_ns"].to_numpy(zero_copy_only=False).astype(np.int64),
        **{name: table[name].to_numpy(zero_copy_only=False).astype(np.float64) for name in names},
    }


def inspect_h5(episode_dir: Path) -> dict[str, Any]:
    path = episode_dir / "trajectory.h5"
    if not path.is_file():
        return {"readable": False, "finite": False, "count": 0, "fields": {}}
    fields: dict[str, dict[str, Any]] = {}
    finite = True
    try:
        with h5py.File(path, "r") as handle:
            attrs = {key: _json_scalar(value) for key, value in handle.attrs.items()}

            def visitor(name: str, obj: Any) -> None:
                nonlocal finite
                if isinstance(obj, h5py.Dataset):
                    fields[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
                    if obj.dtype.kind in "fiu":
                        finite = finite and bool(np.isfinite(np.asarray(obj)).all())

            handle.visititems(visitor)
            count = int(handle["observation/timestamp/monotonic_ns"].shape[0])
    except Exception as exc:
        return {"readable": False, "finite": False, "count": 0, "fields": {}, "error": str(exc)}
    return {"readable": True, "finite": finite, "count": count, "fields": fields, "attrs": attrs}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def inspect_video(path: Path, *, decode_ends: bool = True) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "decodable": False, "frame_count": 0, "error": "missing"}
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,time_base",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        return {
            "path": str(path),
            "decodable": False,
            "frame_count": 0,
            "error": result.stderr.strip(),
        }
    stream = (json.loads(result.stdout).get("streams") or [{}])[0]
    fps = _fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    report = {
        "path": str(path),
        "decodable": True,
        "frame_count": int(stream.get("nb_frames") or 0),
        "fps": fps,
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration_seconds": float(stream.get("duration") or 0.0),
        "time_base": stream.get("time_base"),
    }
    if decode_ends:
        try:
            import cv2

            capture = cv2.VideoCapture(str(path))
            first_ok, _ = capture.read()
            if report["frame_count"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, report["frame_count"] - 1)
            last_ok, _ = capture.read()
            capture.release()
            report["decodable"] = bool(first_ok and last_ok)
        except Exception as exc:
            report["decodable"] = False
            report["error"] = str(exc)
    return report


def _fraction(value: str) -> float:
    numerator, _, denominator = value.partition("/")
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def raw_signature(episode_dir: Path) -> str:
    relative_paths = [
        "trajectory.h5",
        "robot_telemetry.npz",
        "tactile/ati_raw.parquet",
        "tactile/gelsight_left.mp4",
        "tactile/gelsight_left_timestamps.npy",
        "recordings/MP4/exterior_image_1_left.mp4",
        "recordings/MP4/wrist_image_left.mp4",
        "recordings/timestamps/exterior_image_1_left.npz",
        "recordings/timestamps/wrist_image_left.npz",
    ]
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = episode_dir / relative
        if path.is_file():
            stat = path.stat()
            digest.update(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        else:
            digest.update(f"{relative}:missing\n".encode())
    return digest.hexdigest()


def video_source(episode_dir: Path, stream: str) -> Path:
    return episode_dir / VIDEO_STREAM_PATHS[stream]
