"""Atomic Fabric-DROID episode recorder."""

from __future__ import annotations

import os
import select
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fabric_droid.capture.safety import assert_no_robot_motion, preflight_output
from fabric_droid.io_utils import atomic_write_json
from fabric_droid.schemas import (
    CANONICAL_TASK_INSTRUCTION,
    EVENT_NAMES,
    SCHEMA_VERSION,
    EpisodeMetadata,
    EventMarker,
    ForceCalibration,
)
from fabric_droid.sensors.ati import ATIStream
from fabric_droid.sensors.base import SensorStream
from fabric_droid.sensors.camera import CameraSpec, CameraStream
from fabric_droid.sensors.synthetic import SyntheticATIStream, SyntheticCameraStream
from fabric_droid.sync.clock import analyze_timestamps


@dataclass(frozen=True)
class RecorderOptions:
    output_dir: Path
    duration_sec: float = 10.0
    dry_run: bool = False
    robot_disabled: bool = True
    record_only: bool = True
    d435_serial: str = ""
    wrist_kind: str = "uvc"
    wrist_source: str = "0"
    wrist_serial: str = "wrist-uvc"
    gelsight_source: str = "1"
    gelsight_serial: str = "gelsight-left"
    ati_enabled: bool = True
    ati_endpoint: str = "tcp://192.168.1.20:5555"
    width: int = 320
    height: int = 180
    fps: float = 15.0
    gelsight_fps: float = 30.0
    minimum_free_gib: float = 0.25
    sensor_warmup_timeout_sec: float = 5.0


def _write_video_atomic(path: Path, frames: Iterable[Any], fps: float) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to save RGB video") from exc
    frame_list = list(frames)
    if not frame_list:
        raise RuntimeError(f"cannot write empty video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    height, width = frame_list[0].image_bgr.shape[:2]
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"unable to open video writer: {temporary}")
    try:
        for frame in frame_list:
            writer.write(frame.image_bgr)
    finally:
        writer.release()
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_ati_parquet_atomic(path: Path, samples: list[Any]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to preserve ATI data as Parquet") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    schema = pa.schema(
        [
            ("timestamp_monotonic_ns", pa.int64()),
            ("timestamp_wall_ns", pa.int64()),
            ("sample_index", pa.int64()),
            ("Fx", pa.float64()),
            ("Fy", pa.float64()),
            ("Fz", pa.float64()),
            ("Tx", pa.float64()),
            ("Ty", pa.float64()),
            ("Tz", pa.float64()),
            ("device_status", pa.string()),
            ("device_timestamp_ns", pa.int64()),
        ]
    )
    rows = [
        {
            "timestamp_monotonic_ns": sample.timestamp_monotonic_ns,
            "timestamp_wall_ns": sample.timestamp_wall_ns,
            "sample_index": sample.sample_index,
            "Fx": sample.fx,
            "Fy": sample.fy,
            "Fz": sample.fz,
            "Tx": sample.tx,
            "Ty": sample.ty,
            "Tz": sample.tz,
            "device_status": sample.device_status,
            "device_timestamp_ns": sample.device_timestamp_ns,
        }
        for sample in samples
    ]
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


class FabricEpisodeRecorder:
    def __init__(
        self,
        options: RecorderOptions,
        metadata: EpisodeMetadata,
        calibration: ForceCalibration | None = None,
    ) -> None:
        self.options = options
        self.metadata = metadata
        self.calibration = calibration or ForceCalibration(calibration_id=metadata.calibration_id)
        self.events: list[EventMarker] = []
        self.streams: list[SensorStream] = []
        self._started_ns: int | None = None
        self._stopped_ns: int | None = None
        self._robot_samples: list[dict[str, Any]] = []
        self._episode_partial = options.output_dir / f".{metadata.episode_id}.inprogress"
        self.episode_dir = options.output_dir / metadata.episode_id

    def _make_streams(self) -> list[SensorStream]:
        camera_specs = [
            CameraSpec(
                "exterior_image_1_left",
                "d435",
                self.options.d435_serial,
                self.options.d435_serial or "d435",
                self.options.width,
                self.options.height,
                self.options.fps,
            ),
            CameraSpec(
                "wrist_image_left",
                self.options.wrist_kind,
                self.options.wrist_serial if self.options.wrist_kind == "d435" else self.options.wrist_source,
                self.options.wrist_serial,
                self.options.width,
                self.options.height,
                self.options.fps,
            ),
            CameraSpec(
                "gelsight_left",
                "uvc",
                self.options.gelsight_source,
                self.options.gelsight_serial,
                self.options.width,
                self.options.height,
                self.options.gelsight_fps,
            ),
        ]
        if self.options.dry_run:
            streams: list[SensorStream] = [*(SyntheticCameraStream(spec) for spec in camera_specs)]
            if self.options.ati_enabled:
                streams.append(SyntheticATIStream(500.0))
            return streams
        streams = [*(CameraStream(spec) for spec in camera_specs)]
        if self.options.ati_enabled:
            streams.append(ATIStream(self.options.ati_endpoint))
        return streams

    def preflight(self) -> dict[str, Any]:
        assert_no_robot_motion(self.options.robot_disabled, self.options.record_only, self.options.dry_run)
        if self.episode_dir.exists() or self._episode_partial.exists():
            raise FileExistsError(f"episode already exists: {self.metadata.episode_id}")
        return {
            **preflight_output(self.options.output_dir, self.options.minimum_free_gib),
            "robot_motion_enabled": False,
            "normal_force_feedback_enabled": False,
            "normal_force_calibrated": self.calibration.normal_force_calibrated,
            "ati_enabled": self.options.ati_enabled,
            "ati_endpoint": self.options.ati_endpoint if self.options.ati_enabled else None,
        }

    def mark_event(self, name: str, source: str = "manual", confidence: float = 1.0) -> None:
        self.events.append(EventMarker(name, time.monotonic_ns(), source=source, confidence=confidence))

    def start(self) -> None:
        preflight = self.preflight()
        self._episode_partial.mkdir(parents=True)
        atomic_write_json(self._episode_partial / "preflight.json", preflight)
        self.streams = self._make_streams()
        try:
            for stream in self.streams:
                stream.start()
            deadline = time.monotonic() + self.options.sensor_warmup_timeout_sec
            while time.monotonic() < deadline:
                failed = [stream for stream in self.streams if stream.error is not None]
                if failed:
                    raise RuntimeError(f"sensor warmup failed: {failed[0].name}") from failed[0].error
                if all(stream.snapshot().get("healthy", False) for stream in self.streams):
                    break
                time.sleep(0.02)
            else:
                unhealthy = [stream.name for stream in self.streams if not stream.snapshot().get("healthy", False)]
                raise RuntimeError(f"sensor warmup timeout: {unhealthy}")
            self._started_ns = time.monotonic_ns()
        except Exception as exc:
            for stream in reversed(self.streams):
                try:
                    stream.stop()
                except Exception:
                    pass
            atomic_write_json(
                self._episode_partial / "CAPTURE_INCOMPLETE.json",
                {"error": str(exc), "timestamp_monotonic_ns": time.monotonic_ns(), "phase": "sensor_warmup"},
            )
            raise

    def _append_synthetic_robot_sample(self, elapsed: float) -> None:
        index = len(self._robot_samples)
        joint = np.asarray([0.1 * np.sin(elapsed + axis) for axis in range(7)], dtype=np.float64)
        joint_velocity = np.asarray([0.1 * np.cos(elapsed + axis) for axis in range(7)], dtype=np.float64)
        progress = elapsed / max(self.options.duration_sec, 1e-6)
        if progress < 0.12:
            gripper = 0.075
        elif progress < 0.20:
            gripper = 0.025
        elif progress < 0.28:
            gripper = 0.060
        elif progress < 0.36:
            gripper = 0.025
        elif progress < 0.44:
            gripper = 0.060
        elif progress < 0.90:
            gripper = 0.025
        else:
            gripper = 0.075
        self._robot_samples.append(
            {
                "timestamp_monotonic_ns": time.monotonic_ns(),
                "joint_positions": joint,
                "gripper_position": gripper,
                "joint_velocity": joint_velocity,
                "commanded_gripper_position": gripper,
                "sample_index": index,
            }
        )

    def _keyboard_event_loop(self, stop_event: threading.Event) -> None:
        shortcuts = {"p": "probe_complete", "r": "release_time"}
        while not stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            command = sys.stdin.readline().strip()
            if not command:
                continue
            name = shortcuts.get(command.lower(), command)
            if name not in EVENT_NAMES:
                print(f"ignored unknown event '{command}'", file=sys.stderr)
                continue
            self.mark_event(name, source="manual")

    def record_for_duration(self, *, interactive_events: bool = False) -> Path:
        self.start()
        assert self._started_ns is not None
        event_offsets = (
            dict(zip(EVENT_NAMES, np.linspace(0.05, 0.98, len(EVENT_NAMES)), strict=True))
            if self.options.dry_run
            else {}
        )
        pending = list(EVENT_NAMES) if self.options.dry_run else []
        next_robot_ns = self._started_ns
        end_ns = self._started_ns + int(self.options.duration_sec * 1e9)
        keyboard_stop = threading.Event()
        keyboard_thread = None
        if interactive_events:
            keyboard_thread = threading.Thread(
                target=self._keyboard_event_loop,
                args=(keyboard_stop,),
                name="fabric-droid-event-input",
                daemon=True,
            )
            keyboard_thread.start()

        def stop_keyboard() -> None:
            keyboard_stop.set()
            if keyboard_thread is not None:
                keyboard_thread.join(timeout=1.0)

        try:
            while time.monotonic_ns() < end_ns:
                now_ns = time.monotonic_ns()
                elapsed = (now_ns - self._started_ns) / 1e9
                fraction = elapsed / self.options.duration_sec
                while pending and fraction >= event_offsets[pending[0]]:
                    self.mark_event(pending.pop(0), source="synthetic" if self.options.dry_run else "automatic")
                if self.options.dry_run and now_ns >= next_robot_ns:
                    self._append_synthetic_robot_sample(elapsed)
                    next_robot_ns += int(1e9 / 15.0)
                failed = [stream for stream in self.streams if stream.error is not None]
                if failed:
                    raise RuntimeError(f"sensor stream failed: {failed[0].name}") from failed[0].error
                time.sleep(0.002)
            if pending and pending == ["episode_end"]:
                self.mark_event("episode_end", source="synthetic" if self.options.dry_run else "automatic")
                pending.clear()
            if pending:
                raise RuntimeError(f"duration too short to emit required events: {pending}")
            if not self.options.dry_run:
                missing = {"probe_complete", "release_time"} - {event.name for event in self.events}
                if missing:
                    raise RuntimeError(f"missing required manual events: {sorted(missing)}")
            stop_keyboard()
            return self.close(success=True)
        except Exception as exc:
            stop_keyboard()
            self.close(success=False, failure_reason=str(exc))
            raise

    def close(self, success: bool, failure_reason: str = "") -> Path:
        errors: list[str] = []
        for stream in reversed(self.streams):
            try:
                stream.stop()
            except Exception as exc:
                errors.append(f"{stream.name}: {exc}")
        self._stopped_ns = time.monotonic_ns()
        if self._episode_partial.exists():
            try:
                self._persist(success=success and not errors, failure_reason=failure_reason or "; ".join(errors))
            except Exception as exc:
                atomic_write_json(
                    self._episode_partial / "CAPTURE_INCOMPLETE.json",
                    {"error": str(exc), "timestamp_monotonic_ns": time.monotonic_ns()},
                )
                raise
            if success and not errors:
                os.replace(self._episode_partial, self.episode_dir)
                return self.episode_dir
        return self._episode_partial

    def _persist(self, success: bool, failure_reason: str) -> None:
        assert self._started_ns is not None and self._stopped_ns is not None
        camera_streams = {stream.name: stream for stream in self.streams if hasattr(stream, "frames")}
        ati_stream = next((stream for stream in self.streams if hasattr(stream, "samples")), None)
        camera_frames = {
            name: [
                frame for frame in stream.frames if self._started_ns <= frame.timestamp_monotonic_ns <= self._stopped_ns
            ]
            for name, stream in camera_streams.items()
        }
        ati_samples = (
            [
                sample
                for sample in ati_stream.samples
                if self._started_ns <= sample.timestamp_monotonic_ns <= self._stopped_ns
            ]
            if ati_stream is not None
            else []
        )
        recordings = self._episode_partial / "recordings"
        tactile = self._episode_partial / "tactile"
        for name in ("exterior_image_1_left", "wrist_image_left"):
            stream = camera_streams[name]
            frames = camera_frames[name]
            _write_video_atomic(recordings / "MP4" / f"{name}.mp4", frames, stream.spec.fps)
            _write_npz_atomic(
                recordings / "timestamps" / f"{name}.npz",
                frame_index=np.asarray([frame.frame_index for frame in frames], dtype=np.int64),
                timestamp_monotonic_ns=np.asarray([frame.timestamp_monotonic_ns for frame in frames], dtype=np.int64),
                frame_received_timestamp_ns=np.asarray(
                    [frame.frame_received_timestamp_ns for frame in frames], dtype=np.int64
                ),
                device_timestamp_ns=np.asarray([frame.device_timestamp_ns for frame in frames], dtype=np.int64),
            )
        gelsight = camera_streams["gelsight_left"]
        gelsight_frames = camera_frames["gelsight_left"]
        _write_video_atomic(tactile / "gelsight_left.mp4", gelsight_frames, gelsight.spec.fps)
        gelsight_timestamps = np.asarray([frame.timestamp_monotonic_ns for frame in gelsight_frames], dtype=np.int64)
        _write_npy_atomic(tactile / "gelsight_left_timestamps.npy", gelsight_timestamps)
        _write_npz_atomic(
            tactile / "gelsight_left_frame_metadata.npz",
            frame_index=np.asarray([frame.frame_index for frame in gelsight_frames], dtype=np.int64),
            timestamp_monotonic_ns=gelsight_timestamps,
            frame_received_timestamp_ns=np.asarray(
                [frame.frame_received_timestamp_ns for frame in gelsight_frames], dtype=np.int64
            ),
            device_timestamp_ns=np.asarray([frame.device_timestamp_ns for frame in gelsight_frames], dtype=np.int64),
            dropped_frame_count=np.asarray([gelsight.dropped_frame_count], dtype=np.int64),
        )
        _write_ati_parquet_atomic(tactile / "ati_raw.parquet", ati_samples)
        atomic_write_json(
            tactile / "events.json",
            {"schema_version": SCHEMA_VERSION, "events": [event.to_dict() for event in self.events]},
        )
        atomic_write_json(tactile / "calibration.json", self.calibration.to_dict())
        metadata = {**self.metadata.to_dict(), "success": success, "failure_reason": failure_reason}
        atomic_write_json(self._episode_partial / f"metadata_{self.metadata.episode_id}.json", metadata)
        if self.options.dry_run:
            self._write_trajectory_h5(success, failure_reason)
        report = {
            "schema_version": SCHEMA_VERSION,
            "complete": success,
            "failure_reason": failure_reason,
            "episode_start_monotonic_ns": self._started_ns,
            "episode_end_monotonic_ns": self._stopped_ns,
            "duration_sec": (self._stopped_ns - self._started_ns) / 1e9,
            "robot_motion_enabled": False,
            "normal_force_feedback_enabled": False,
            "ati_enabled": self.options.ati_enabled,
            "ati_connected": ati_stream is not None and bool(ati_samples),
            "streams": {stream.name: stream.snapshot() for stream in self.streams},
            "clock_reports": {
                name: analyze_timestamps([frame.timestamp_monotonic_ns for frame in camera_frames[name]]).to_dict()
                for name in camera_streams
            }
            | {
                "ati_nano17": {
                    **analyze_timestamps([sample.timestamp_monotonic_ns for sample in ati_samples]).to_dict(),
                    "enabled": self.options.ati_enabled,
                    "connected": ati_stream is not None and bool(ati_samples),
                }
            },
        }
        atomic_write_json(tactile / "capture_report.json", report)
        if success:
            atomic_write_json(
                self._episode_partial / "COMPLETE.json",
                {"complete": True, "closed_monotonic_ns": self._stopped_ns, "schema_version": SCHEMA_VERSION},
            )

    def _write_trajectory_h5(self, success: bool, failure_reason: str) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required for synthetic DROID trajectory output") from exc
        path = self._episode_partial / "trajectory.h5"
        temporary = path.with_name(".trajectory.h5.tmp")
        samples = self._robot_samples
        if len(samples) < 2:
            raise RuntimeError("synthetic robot stream contains too few samples")
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                {
                    "success": success,
                    "failure": not success,
                    "failure_reason": failure_reason,
                    "robot_motion_enabled": False,
                    "time_axis": "host_monotonic_ns",
                }
            )
            observation = handle.create_group("observation")
            robot = observation.create_group("robot_state")
            robot.create_dataset("joint_positions", data=np.stack([sample["joint_positions"] for sample in samples]))
            robot.create_dataset("gripper_position", data=np.asarray([sample["gripper_position"] for sample in samples]))
            timestamp = observation.create_group("timestamp")
            timestamp.create_dataset(
                "monotonic_ns", data=np.asarray([sample["timestamp_monotonic_ns"] for sample in samples], dtype=np.int64)
            )
            action = handle.create_group("action")
            action.create_dataset("joint_velocity", data=np.stack([sample["joint_velocity"] for sample in samples]))
            action.create_dataset(
                "gripper_position",
                data=np.asarray([sample["commanded_gripper_position"] for sample in samples]),
            )
        os.replace(temporary, path)


def default_metadata(
    destination_tray: str,
    swatch_uid: str,
    split: str,
    session_id: str | None = None,
    episode_id: str | None = None,
) -> EpisodeMetadata:
    episode_id = episode_id or f"episode_{uuid.uuid4().hex[:12]}"
    session_id = session_id or f"session_{time.strftime('%Y%m%d')}"
    if destination_tray != "target_tray":
        raise ValueError(
            "destination_tray must be 'target_tray'; color/direction labels "
            "are no longer supported for new captures"
        )
    prompt = CANONICAL_TASK_INSTRUCTION
    return EpisodeMetadata(
        episode_id=episode_id,
        task_instruction=prompt,
        destination_tray=destination_tray,
        source_slot="source_slot_1",
        swatch_uid=swatch_uid,
        split=split,
        start_pose_bucket="ready",
        success=True,
        failure_reason="",
        operator="unknown",
        session_id=session_id,
        robot_id="robot-disabled",
        camera_serials={"exterior_image_1_left": "d435", "wrist_image_left": "wrist-uvc"},
        gelsight_serial="gelsight-left",
        ati_serial="ati-nano17",
        calibration_id="uncalibrated",
        software_git_commits={"droid": "working-tree"},
        robot_motion_enabled=False,
    )
