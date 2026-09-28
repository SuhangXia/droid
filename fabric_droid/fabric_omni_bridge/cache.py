"""Offline Stage-3 physical-state cache generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from fabric_droid.fabric_omni_bridge.encoder import FabricPhysicalEncoder
from fabric_droid.io_utils import atomic_write_json
from fabric_droid.schemas import ForceCalibration


def _video_frames(path: Path) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to decode {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame[..., ::-1].copy())
    capture.release()
    return frames


def cache_episode(
    encoder: FabricPhysicalEncoder,
    episode_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if encoder.fabric_omni_root == output_dir.resolve() or encoder.fabric_omni_root in output_dir.resolve().parents:
        raise ValueError("physical-state cache output must not be inside Fabric-Omni")
    metadata_path = next(iter(episode_dir.glob("metadata_*.json")))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    events = json.loads((episode_dir / "tactile" / "events.json").read_text(encoding="utf-8"))["events"]
    probe_ns = next(int(event["timestamp_monotonic_ns"]) for event in events if event["name"] == "probe_complete")
    frames = _video_frames(episode_dir / "tactile" / "gelsight_left.mp4")
    timestamps_ns = np.load(episode_dir / "tactile" / "gelsight_left_timestamps.npy")
    end = int(np.searchsorted(timestamps_ns, probe_ns, side="right"))
    start = max(0, end - 48)
    if end - start < 2:
        raise ValueError("too few GelSight frames before probe_complete")
    import pyarrow.parquet as pq

    ati = pq.read_table(episode_dir / "tactile" / "ati_raw.parquet")
    ati_ns = np.asarray(ati["timestamp_monotonic_ns"], dtype=np.int64)
    ati_start_ns = int(timestamps_ns[start])
    ati_mask = (ati_ns >= ati_start_ns) & (ati_ns <= probe_ns)
    wrench = np.stack([np.asarray(ati[name], dtype=np.float32) for name in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")], axis=1)
    calibration_payload = json.loads((episode_dir / "tactile" / "calibration.json").read_text(encoding="utf-8"))
    calibration = ForceCalibration(
        calibration_id=calibration_payload["calibration_id"],
        T_ati_to_gripper=calibration_payload.get("T_ati_to_gripper"),
        gripper_normal_axis=calibration_payload.get("gripper_normal_axis"),
        force_sign=calibration_payload.get("force_sign"),
        bias_wrench=calibration_payload.get("bias_wrench", [0.0] * 6),
    )
    state = encoder.encode_policy_state(
        np.stack(frames[start:end]),
        wrench[ati_mask],
        (),
        calibration=calibration,
        frame_timestamps_sec=timestamps_ns[start:end] / 1e9,
        ati_timestamps_sec=ati_ns[ati_mask] / 1e9,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metadata['episode_id']}.physical_state.npz"
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            global_tactile_latent=state.global_tactile_latent,
            predicted_normal_force=np.float32(state.predicted_normal_force),
            contact_probability=np.float32(state.contact_probability),
            contact_phase=np.int64(state.contact_phase),
            contact_area_proxy=np.float32(state.contact_area_proxy),
            softness_score=np.float32(state.softness_score),
            uncertainty=np.float32(state.uncertainty),
            tactile_valid=np.bool_(state.tactile_valid),
            force_valid=np.bool_(state.force_valid),
            physical_state_valid=np.bool_(state.physical_state_valid),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_path)
    manifest = {
        **state.serializable_metadata(),
        "episode_id": metadata["episode_id"],
        "swatch_uid": metadata["swatch_uid"],
        "split": metadata["split"],
        "source_episode": str(episode_dir.resolve()),
        "cache_file": str(output_path),
        "normalization_fitted": False,
        "heldout_used_for_fitting": False,
    }
    atomic_write_json(output_path.with_suffix(".json"), manifest)
    return manifest
