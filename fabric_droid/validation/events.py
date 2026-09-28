"""Generate candidate contact/release events from observable signals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _crossings(values: np.ndarray, threshold: float, rising: bool) -> np.ndarray:
    if values.size < 2:
        return np.asarray([], dtype=np.int64)
    if rising:
        return np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold)) + 1
    return np.flatnonzero((values[:-1] >= threshold) & (values[1:] < threshold)) + 1


def detect_event_candidates(episode_dir: Path) -> dict[str, Any]:
    import cv2
    import h5py
    import pyarrow.parquet as pq

    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        robot_ns = np.asarray(handle["observation/timestamp/monotonic_ns"], dtype=np.int64)
        gripper = np.asarray(handle["observation/robot_state/gripper_position"], dtype=np.float64)
    ati = pq.read_table(episode_dir / "tactile" / "ati_raw.parquet")
    ati_ns = np.asarray(ati["timestamp_monotonic_ns"], dtype=np.int64)
    raw_force = np.sqrt(sum(np.asarray(ati[name], dtype=np.float64) ** 2 for name in ("Fx", "Fy", "Fz")))
    baseline_count = max(10, min(raw_force.size // 10, 500))
    baseline = float(np.median(raw_force[:baseline_count]))
    deviation = np.abs(raw_force - baseline)
    force_threshold = float(np.median(deviation) + 6.0 * max(np.median(np.abs(deviation - np.median(deviation))), 1e-4))
    force_contact = _crossings(deviation, force_threshold, True)
    force_release = _crossings(deviation, force_threshold * 0.5, False)

    video = cv2.VideoCapture(str(episode_dir / "tactile" / "gelsight_left.mp4"))
    deltas: list[float] = []
    previous = None
    while True:
        ok, frame = video.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        deltas.append(0.0 if previous is None else float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
    video.release()
    tactile_ns = np.load(episode_dir / "tactile" / "gelsight_left_timestamps.npy")
    delta_values = np.asarray(deltas, dtype=np.float64)
    tactile_threshold = float(np.median(delta_values) + 4.0 * np.std(delta_values))
    tactile_changes = np.flatnonzero(delta_values >= tactile_threshold)

    derivative = np.gradient(gripper, robot_ns / 1e9)
    close_candidates = np.flatnonzero(derivative < -max(0.002, float(np.std(derivative) * 2)))
    open_candidates = np.flatnonzero(derivative > max(0.002, float(np.std(derivative) * 2)))
    manual_payload = json.loads((episode_dir / "tactile" / "events.json").read_text(encoding="utf-8"))
    manual = {event["name"]: int(event["timestamp_monotonic_ns"]) for event in manual_payload["events"]}

    def times(indices: np.ndarray, timestamps: np.ndarray, limit: int = 12) -> list[int]:
        return [int(value) for value in timestamps[indices[:limit]]] if indices.size else []

    candidates = {
        "force_contact_monotonic_ns": times(force_contact, ati_ns),
        "force_release_monotonic_ns": times(force_release, ati_ns),
        "gripper_closing_monotonic_ns": times(close_candidates, robot_ns),
        "gripper_opening_monotonic_ns": times(open_candidates, robot_ns),
        "gelsight_delta_monotonic_ns": times(tactile_changes, tactile_ns),
    }
    comparisons: dict[str, Any] = {}
    combined = sorted(value for values in candidates.values() for value in values)
    for name in ("probe_complete", "release_time"):
        if name in manual and combined:
            nearest = min(combined, key=lambda value: abs(value - manual[name]))
            comparisons[name] = {
                "manual_monotonic_ns": manual[name],
                "nearest_candidate_monotonic_ns": nearest,
                "absolute_delta_ms": abs(nearest - manual[name]) / 1e6,
            }
    return {
        "automatic_only": True,
        "normal_force_used": False,
        "raw_force_norm_used": True,
        "warning": "Raw force norm is used only for event candidates, never as calibrated normal force.",
        "thresholds": {
            "raw_force_deviation": force_threshold,
            "gelsight_mean_absolute_delta": tactile_threshold,
        },
        "candidates": candidates,
        "manual_comparison": comparisons,
    }
