"""Offline Segment A/B manifests retaining absolute source timestamps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fabric_droid.io_utils import atomic_write_json


def _load_source(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, Any]]:
    import h5py

    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        timestamps = np.asarray(handle["observation/timestamp/monotonic_ns"], dtype=np.int64)
        gripper = np.asarray(handle["observation/robot_state/gripper_position"], dtype=np.float64)
    events_payload = json.loads((episode_dir / "tactile" / "events.json").read_text(encoding="utf-8"))
    events = {event["name"]: int(event["timestamp_monotonic_ns"]) for event in events_payload["events"]}
    metadata_path = next(iter(episode_dir.glob("metadata_*.json")))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return timestamps, gripper, events, metadata


def build_segment_manifest(episode_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    timestamps, gripper, events, metadata = _load_source(episode_dir)
    for name in ("probe_complete", "release_time"):
        if name not in events:
            raise ValueError(f"cannot segment episode without {name}")
    probe_index = int(np.searchsorted(timestamps, events["probe_complete"], side="left"))
    release_index = int(np.searchsorted(timestamps, events["release_time"], side="right") - 1)
    probe_index = min(max(probe_index, 1), timestamps.size - 2)
    release_index = min(max(release_index, probe_index + 1), timestamps.size - 1)
    initial_width = float(np.median(gripper[: min(10, probe_index)]))
    held_width = float(gripper[probe_index])
    cut_valid = held_width < initial_width - 1e-4
    destination_prompt = "Place the held fabric in the target tray."

    def segment(name: str, prompt: str, start: int, end: int) -> dict[str, Any]:
        return {
            "segment_name": name,
            "prompt": prompt,
            "source_episode_id": metadata["episode_id"],
            "source_episode_path": str(episode_dir.resolve()),
            "source_start_frame": start,
            "source_end_frame_inclusive": end,
            "frame_count": end - start + 1,
            "timestep_start": 0,
            "absolute_start_monotonic_ns": int(timestamps[start]),
            "absolute_end_monotonic_ns": int(timestamps[end]),
            "split": metadata["split"],
            "swatch_uid": metadata["swatch_uid"],
        }

    manifest = {
        "source_episode_id": metadata["episode_id"],
        "valid": cut_valid,
        "failures": [] if cut_valid else ["Segment B does not start in a clearly held-gripper state"],
        "probe_complete_event_ns": events["probe_complete"],
        "release_time_event_ns": events["release_time"],
        "probe_complete_frame_error_ms": abs(int(timestamps[probe_index]) - events["probe_complete"]) / 1e6,
        "release_time_frame_error_ms": abs(int(timestamps[release_index]) - events["release_time"]) / 1e6,
        "gripper_initial_median": initial_width,
        "gripper_at_segment_b_start": held_width,
        "segments": [
            segment(
                "segment_a_probe",
                "Inspect the fabric and hold it securely.",
                0,
                probe_index,
            ),
            segment("segment_b_place", destination_prompt, probe_index, release_index),
        ],
    }
    output_path = output_path or episode_dir / "segments.json"
    atomic_write_json(output_path, manifest)
    return manifest
