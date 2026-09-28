"""Streaming sensor-sidecar recorder used by the desktop collection UI."""

from __future__ import annotations

import csv
import os
import queue
import shutil
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fabric_droid.capture.recorder import (
    _write_ati_parquet_atomic,
    _write_npy_atomic,
    _write_npz_atomic,
    default_metadata,
)
from fabric_droid.capture.robot_trajectory import write_droid_trajectory_h5
from fabric_droid.io_utils import atomic_write_json, git_commit
from fabric_droid.schemas import SCHEMA_VERSION, EventMarker, ForceCalibration
from fabric_droid.sensors.ati import ATIStream
from fabric_droid.sensors.base import SensorStream
from fabric_droid.sensors.camera import CameraFrame, CameraSpec, CameraStream
from fabric_droid.sync.clock import analyze_timestamps


@dataclass(frozen=True)
class CollectionConfig:
    output_root: Path
    episode_id: str
    session_id: str
    swatch_uid: str
    destination_tray: str
    split: str
    operator: str
    wrist_serial: str
    exterior_serial: str
    gelsight_source: str
    gelsight_serial: str
    wrist_kind: str = "d435"
    wrist_source: str = ""
    wrist_width: int = 640
    wrist_height: int = 480
    wrist_fps: float = 30.0
    wrist_pixel_format: str | None = None
    wrist_exposure_scale: float | None = None
    wrist_exposure_absolute: int | None = None
    wrist_white_balance_temperature: int | None = None
    robot_ip: str = "192.168.0.116"
    ati_enabled: bool = True
    ati_endpoint: str = "tcp://192.168.1.20:5555"
    rgb_width: int = 640
    rgb_height: int = 480
    rgb_fps: float = 30.0
    rgb_exposure: float | None = 141.0
    rgb_white_balance: float | None = 3780.0
    rgb_tint: float = -17.0
    gelsight_width: int = 640
    gelsight_height: int = 480
    gelsight_fps: float = 25.0


class _StreamingVideoSink:
    def __init__(
        self,
        path: Path,
        fps: float,
        *,
        background_path: Path | None = None,
        output_size: tuple[int, int] | None = None,
        reencode_measured_fps: bool = False,
        max_queue_frames: int = 120,
    ) -> None:
        if max_queue_frames < 1:
            raise ValueError("max_queue_frames must be at least 1")
        self.path = path
        self.fps = fps
        self.background_path = background_path
        self.output_size = output_size
        self.reencode_measured_fps = reencode_measured_fps
        self.encoded_fps = float(fps)
        self.background_written = False
        self.writer: Any = None
        self.frame_index: list[int] = []
        self.timestamp_monotonic_ns: list[int] = []
        self.frame_received_timestamp_ns: list[int] = []
        self.device_timestamp_ns: list[int] = []
        self.size: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._video_queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=max_queue_frames
        )
        self._writer_thread: threading.Thread | None = None
        self._writer_error: BaseException | None = None
        self._written_count = 0
        self._peak_queue_depth = 0
        self._closed = False

    def append(self, frame: CameraFrame) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to save collection video") from exc
        image_bgr = frame.image_bgr
        if self.output_size is not None:
            source_height, source_width = image_bgr.shape[:2]
            if (source_width, source_height) != self.output_size:
                image_bgr = cv2.resize(
                    image_bgr,
                    self.output_size,
                    interpolation=cv2.INTER_AREA,
                )
        # CameraFrame owns its ndarray and the queue keeps it alive after the
        # rolling preview buffer evicts it. The writer only reads this image,
        # so a second full-frame copy would waste memory bandwidth.
        height, width = image_bgr.shape[:2]
        with self._lock:
            if self._closed:
                raise RuntimeError(f"video sink is already closed: {self.path}")
            if self._writer_error is not None:
                raise RuntimeError(f"video writer failed: {self.path}") from self._writer_error
            if self.size is None:
                self.size = (width, height)
            if self.size != (width, height):
                raise RuntimeError(
                    f"camera resolution changed during capture: {self.size} -> {(width, height)}"
                )
            self._ensure_writer_thread_locked()
        try:
            self._video_queue.put(image_bgr, timeout=2.0)
        except queue.Full as exc:
            raise RuntimeError(
                f"video writer queue stayed full for 2s: {self.path}"
            ) from exc
        with self._lock:
            self._peak_queue_depth = max(
                self._peak_queue_depth,
                self._video_queue.qsize(),
            )
            self.frame_index.append(frame.frame_index)
            self.timestamp_monotonic_ns.append(frame.timestamp_monotonic_ns)
            self.frame_received_timestamp_ns.append(frame.frame_received_timestamp_ns)
            self.device_timestamp_ns.append(frame.device_timestamp_ns)

    def _ensure_writer_thread_locked(self) -> None:
        if self._writer_thread is not None:
            return
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"video-writer-{self.path.stem}",
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        try:
            import cv2

            self.path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                image_bgr = self._video_queue.get()
                if image_bgr is None:
                    break
                if self.writer is None:
                    height, width = image_bgr.shape[:2]
                    self.writer = cv2.VideoWriter(
                        str(self.path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (width, height),
                    )
                    if not self.writer.isOpened():
                        raise RuntimeError(
                            f"unable to open video writer: {self.path}"
                        )
                if self.background_path is not None and not self.background_written:
                    self.background_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(
                        str(self.background_path),
                        image_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    ):
                        raise RuntimeError(
                            f"unable to write GelSight background: {self.background_path}"
                        )
                    self.background_written = True
                self.writer.write(image_bgr)
                with self._lock:
                    self._written_count += 1
        except BaseException as exc:
            with self._lock:
                self._writer_error = exc
        finally:
            if self.writer is not None:
                self.writer.release()
                self.writer = None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.frame_index)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            writer_thread = self._writer_thread
        if writer_thread is not None:
            try:
                self._video_queue.put(None, timeout=5.0)
            except queue.Full as exc:
                raise RuntimeError(
                    f"video writer queue could not drain during close: {self.path}"
                ) from exc
            writer_thread.join(timeout=30.0)
            if writer_thread.is_alive():
                raise RuntimeError(f"video writer did not stop within 30s: {self.path}")
        with self._lock:
            writer_error = self._writer_error
            written_count = self._written_count
        if writer_error is not None:
            raise RuntimeError(f"video writer failed: {self.path}") from writer_error
        if written_count != self.count:
            raise RuntimeError(
                f"video writer frame mismatch for {self.path.name}: "
                f"{written_count} written != {self.count} queued"
            )
        if self.reencode_measured_fps:
            self._reencode_at_measured_fps()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued_frame_count": len(self.frame_index),
                "written_frame_count": self._written_count,
                "pending_queue_frames": self._video_queue.qsize(),
                "peak_queue_frames": self._peak_queue_depth,
                "writer_failed": self._writer_error is not None,
                "configured_fps": self.fps,
                "encoded_fps": self.encoded_fps,
            }

    def _reencode_at_measured_fps(self) -> None:
        clock = self.clock_report()
        measured_fps = float(clock.get("measured_hz", 0.0))
        if (
            self.count < 2
            or measured_fps <= 0
            or abs(measured_fps - self.fps) <= 0.5
            or not self.path.is_file()
            or self.size is None
        ):
            return
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required to correct video fps"
            ) from exc
        capture = cv2.VideoCapture(str(self.path))
        temporary = self.path.with_name(f".{self.path.stem}.measured-fps.mp4")
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            measured_fps,
            self.size,
        )
        if not capture.isOpened() or not writer.isOpened():
            capture.release()
            writer.release()
            raise RuntimeError(
                f"unable to re-encode {self.path.name} at measured {measured_fps:.3f} fps"
            )
        written = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
                written += 1
        finally:
            capture.release()
            writer.release()
        if written != self.count:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"video fps correction frame mismatch: {written} != {self.count}"
            )
        os.replace(temporary, self.path)
        self.encoded_fps = measured_fps

    def clock_report(self) -> dict[str, Any]:
        with self._lock:
            timestamps = list(self.timestamp_monotonic_ns)
        return analyze_timestamps(timestamps).to_dict()


def _wall_iso_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _write_csv_atomic(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class StreamedSensorEpisode:
    """Record synchronized sensors and finalize real Quest telemetry as DROID."""

    def __init__(self, config: CollectionConfig) -> None:
        self.config = config
        self.partial_dir = config.output_root / f".{config.episode_id}.inprogress"
        self.episode_dir = config.output_root / config.episode_id
        self.streams: list[SensorStream] = []
        self.events: list[EventMarker] = []
        self.started_ns: int | None = None
        self.stopped_ns: int | None = None
        self._recording = threading.Event()
        self._sinks: dict[str, _StreamingVideoSink] = {}
        self._final_sensor_health: dict[str, Any] | None = None
        self._sensor_fault_reason = ""
        self._sensor_fault_lock = threading.Lock()

    @property
    def robot_telemetry_path(self) -> Path:
        return self.partial_dir / "robot_telemetry.npz"

    def _camera_callback(self, name: str, frame: CameraFrame) -> None:
        if self._recording.is_set():
            self._sinks[name].append(frame)

    def _camera_specs(self) -> list[CameraSpec]:
        return [
            CameraSpec(
                name="exterior_image_1_left",
                kind="d435",
                source=self.config.exterior_serial,
                serial=self.config.exterior_serial,
                width=self.config.rgb_width,
                height=self.config.rgb_height,
                fps=self.config.rgb_fps,
                exposure=self.config.rgb_exposure,
                white_balance=self.config.rgb_white_balance,
                tint=self.config.rgb_tint,
                require_usb3=True,
            ),
            CameraSpec(
                name="wrist_image_left",
                kind=self.config.wrist_kind,
                source=(
                    self.config.wrist_serial
                    if self.config.wrist_kind == "d435"
                    else self.config.wrist_source or self.config.wrist_serial
                ),
                serial=self.config.wrist_serial,
                width=self.config.wrist_width,
                height=self.config.wrist_height,
                fps=self.config.wrist_fps,
                pixel_format=(
                    self.config.wrist_pixel_format
                    if self.config.wrist_kind == "uvc"
                    else None
                ),
                exposure=(
                    self.config.rgb_exposure
                    if self.config.wrist_kind == "d435"
                    else None
                ),
                white_balance=(
                    self.config.rgb_white_balance
                    if self.config.wrist_kind == "d435"
                    else None
                ),
                tint=(
                    self.config.rgb_tint
                    if self.config.wrist_kind == "d435"
                    else 0.0
                ),
                uvc_exposure_scale=(
                    self.config.wrist_exposure_scale
                    if self.config.wrist_kind == "uvc"
                    else None
                ),
                uvc_exposure_absolute=(
                    self.config.wrist_exposure_absolute
                    if self.config.wrist_kind == "uvc"
                    else None
                ),
                uvc_white_balance_temperature=(
                    self.config.wrist_white_balance_temperature
                    if self.config.wrist_kind == "uvc"
                    else None
                ),
                require_usb3=self.config.wrist_kind == "d435",
            ),
            CameraSpec(
                name="gelsight_left",
                kind="uvc",
                source=self.config.gelsight_source,
                serial=self.config.gelsight_serial,
                # This GelSight UVC firmware only advertises 3280x2464 MJPG.
                # Octopi records that native stream and downsizes on write.
                width=3280,
                height=2464,
                fps=self.config.gelsight_fps,
                pixel_format="MJPG",
            ),
        ]

    def start(self, warmup_timeout_sec: float = 8.0) -> None:
        if self.episode_dir.exists() or self.partial_dir.exists():
            raise FileExistsError(f"episode already exists: {self.config.episode_id}")
        if (
            self.config.wrist_kind == "d435"
            and self.config.wrist_serial == self.config.exterior_serial
        ):
            raise ValueError("wrist and third-person views must use different D435 serials")
        wrist_source = self.config.wrist_source or self.config.wrist_serial
        if (
            self.config.wrist_kind == "uvc"
            and wrist_source == self.config.gelsight_source
        ):
            raise ValueError("wrist fisheye and GelSight must use different UVC devices")
        if not self.config.ati_enabled:
            raise ValueError("ATI is mandatory; refusing to start without ATI enabled")
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self.partial_dir.mkdir()
        atomic_write_json(
            self.partial_dir / "device_assignment.json",
            {
                **asdict(self.config),
                "output_root": str(self.config.output_root),
                "saved_wall_time_ns": time.time_ns(),
            },
        )
        self._sinks = {
            "exterior_image_1_left": _StreamingVideoSink(
                self.partial_dir / "recordings/MP4/exterior_image_1_left.mp4",
                self.config.rgb_fps,
                reencode_measured_fps=True,
            ),
            "wrist_image_left": _StreamingVideoSink(
                self.partial_dir / "recordings/MP4/wrist_image_left.mp4",
                self.config.wrist_fps,
                reencode_measured_fps=True,
            ),
            "gelsight_left": _StreamingVideoSink(
                self.partial_dir / "tactile/gelsight_left.mp4",
                self.config.gelsight_fps,
                background_path=self.partial_dir / "tactile/background.jpg",
                output_size=(
                    self.config.gelsight_width,
                    self.config.gelsight_height,
                ),
                reencode_measured_fps=True,
            ),
        }
        self.streams = [
            CameraStream(
                spec,
                max_buffer_frames=2,
                frame_callback=lambda frame, name=spec.name: self._camera_callback(name, frame),
                # GelSight delivers 3280x2464 MJPEG. Downsize once before the
                # rolling buffer and recorder callback instead of copying a
                # decoded ~24 MB BGR frame and resizing it again in the sink.
                frame_output_size=(
                    (self.config.gelsight_width, self.config.gelsight_height)
                    if spec.name == "gelsight_left"
                    else None
                ),
            )
            for spec in self._camera_specs()
        ]
        self.streams.append(ATIStream(self.config.ati_endpoint))
        try:
            for stream in self.streams:
                stream.start()
            deadline = time.monotonic() + warmup_timeout_sec
            while time.monotonic() < deadline:
                failed = [stream for stream in self.streams if stream.error is not None]
                if failed:
                    raise RuntimeError(f"sensor failed: {failed[0].name}") from failed[0].error
                health = self.sensor_health_report()
                if health["ready"]:
                    break
                time.sleep(0.03)
            else:
                health = self.sensor_health_report()
                raise RuntimeError(
                    "sensor warmup timed out: " + "; ".join(health["reasons"])
                )
            episode_start_ns = time.monotonic_ns()
            for stream in self.streams:
                if isinstance(stream, (CameraStream, ATIStream)):
                    stream.reset_gap_monitor(episode_start_ns)
            self.started_ns = episode_start_ns
            self._recording.set()
            # Do not rely on warmup frames/samples to certify an episode. Wait
            # for every mandatory stream to produce data after started_ns.
            episode_gate_deadline = time.monotonic() + min(
                max(warmup_timeout_sec, 1.0),
                2.0,
            )
            while time.monotonic() < episode_gate_deadline:
                failed = [
                    stream for stream in self.streams if stream.error is not None
                ]
                if failed:
                    raise RuntimeError(
                        f"sensor failed: {failed[0].name}"
                    ) from failed[0].error
                health = self.sensor_health_report()
                if health["ready"]:
                    break
                time.sleep(0.03)
            else:
                health = self.sensor_health_report()
                raise RuntimeError(
                    "episode sensor gate timed out: "
                    + "; ".join(health["reasons"])
                )
        except Exception as exc:
            self._recording.clear()
            cleanup_errors: list[str] = []
            for stream in self.streams:
                stream.request_stop()
            for stream in reversed(self.streams):
                try:
                    stream.stop()
                except Exception as stop_exc:
                    cleanup_errors.append(f"{stream.name}: {stop_exc}")
            for name, sink in self._sinks.items():
                try:
                    sink.close()
                except Exception as sink_exc:
                    cleanup_errors.append(f"{name}: {sink_exc}")
            error = str(exc)
            if cleanup_errors:
                error += "; cleanup: " + "; ".join(cleanup_errors)
            atomic_write_json(
                self.partial_dir / "CAPTURE_INCOMPLETE.json",
                {
                    "error": error,
                    "phase": "warmup",
                    "timestamp_monotonic_ns": time.monotonic_ns(),
                },
            )
            raise

    def mark_event(self, name: str) -> None:
        if not self._recording.is_set():
            raise RuntimeError("no recording is active")
        self.events.append(EventMarker(name, time.monotonic_ns(), source="manual"))

    def camera_streams(self) -> dict[str, CameraStream]:
        return {stream.name: stream for stream in self.streams if isinstance(stream, CameraStream)}

    def ati_stream(self) -> ATIStream | None:
        return next((stream for stream in self.streams if isinstance(stream, ATIStream)), None)

    def sensor_health_report(self, *, max_age_sec: float = 1.0) -> dict[str, Any]:
        """Return the mandatory three-camera + ATI recording gate state."""

        since_monotonic_ns = (
            self.started_ns if self._recording.is_set() else None
        )
        required_cameras = {
            "exterior_image_1_left",
            "wrist_image_left",
            "gelsight_left",
        }
        cameras = self.camera_streams()
        camera_reports: dict[str, dict[str, Any]] = {}
        reasons: list[str] = []
        for name in sorted(required_cameras):
            stream = cameras.get(name)
            if stream is None:
                report = {"ready": False, "reasons": ["stream is missing"]}
            else:
                report = stream.readiness(
                    max_age_sec=max_age_sec,
                    max_gap_sec=(
                        max_age_sec
                        if since_monotonic_ns is not None
                        else None
                    ),
                    since_monotonic_ns=since_monotonic_ns,
                )
            camera_reports[name] = report
            reasons.extend(f"{name}: {reason}" for reason in report["reasons"])
        ati = self.ati_stream()
        if ati is None:
            ati_report: dict[str, Any] = {
                "ready": False,
                "reasons": ["stream is missing"],
            }
        else:
            ati_report = ati.readiness(
                max_age_sec=max_age_sec,
                max_gap_sec=(
                    max_age_sec
                    if since_monotonic_ns is not None
                    else None
                ),
                since_monotonic_ns=since_monotonic_ns,
            )
        reasons.extend(f"ati_nano17: {reason}" for reason in ati_report["reasons"])
        sink_reports = {
            name: sink.status() for name, sink in self._sinks.items()
        }
        if self._recording.is_set():
            for name in sorted(required_cameras - sink_reports.keys()):
                reasons.append(f"{name}: video sink is missing")
        for name, report in sink_reports.items():
            if report.get("writer_failed"):
                reasons.append(f"{name}: video writer failed")
            if (
                self._recording.is_set()
                and int(report.get("queued_frame_count", 0)) < 2
            ):
                reasons.append(
                    f"{name}: fewer than 2 frames reached the video sink"
                )
        return {
            "ready": not reasons,
            "reasons": reasons,
            "cameras": camera_reports,
            "ati": ati_report,
            "video_sinks": sink_reports,
            "since_monotonic_ns": since_monotonic_ns,
        }

    def latch_sensor_fault(self, reason: str) -> None:
        """Permanently downgrade this episode after a mandatory sensor fault."""

        normalized = str(reason).strip() or "mandatory sensor gate failed"
        with self._sensor_fault_lock:
            if not self._sensor_fault_reason:
                self._sensor_fault_reason = normalized

    @property
    def sensor_fault_reason(self) -> str:
        with self._sensor_fault_lock:
            return self._sensor_fault_reason

    def stop(self, *, success: bool = True, failure_reason: str = "") -> Path:
        final_health = self.sensor_health_report()
        self._final_sensor_health = final_health
        self._recording.clear()
        self.stopped_ns = time.monotonic_ns()
        stop_errors: list[str] = []
        if self.sensor_fault_reason:
            stop_errors.append(self.sensor_fault_reason)
        if success and not final_health["ready"]:
            stop_errors.append(
                "mandatory sensor gate failed at stop: "
                + "; ".join(final_health["reasons"])
            )
        # Let all sensor workers enter their worker-owned cleanup concurrently.
        # In particular, OpenCV VideoCapture must not be released from this
        # QThreadPool task while its camera thread is inside read().
        for stream in self.streams:
            stream.request_stop()
        for stream in reversed(self.streams):
            try:
                stream.stop()
            except Exception as exc:
                stop_errors.append(f"{stream.name}: {exc}")
        for name, sink in self._sinks.items():
            try:
                sink.close()
            except Exception as exc:
                stop_errors.append(f"{name} video sink: {exc}")
        empty = [name for name, sink in self._sinks.items() if sink.count == 0]
        if empty:
            stop_errors.append(f"empty camera recordings: {empty}")
        final_success = success and not stop_errors
        reason_parts: list[str] = []
        for value in (failure_reason.strip(), *stop_errors):
            if value and value not in reason_parts:
                reason_parts.append(value)
        reason = "; ".join(reason_parts)
        try:
            self._persist(final_success, reason)
        except Exception as exc:
            atomic_write_json(
                self.partial_dir / "CAPTURE_INCOMPLETE.json",
                {"error": str(exc), "phase": "persist", "timestamp_monotonic_ns": time.monotonic_ns()},
            )
            raise
        if final_success:
            os.replace(self.partial_dir, self.episode_dir)
            return self.episode_dir
        return self.partial_dir

    def _persist(self, success: bool, failure_reason: str) -> None:
        if self.started_ns is None or self.stopped_ns is None:
            raise RuntimeError("capture was not started")
        for name in ("exterior_image_1_left", "wrist_image_left"):
            sink = self._sinks[name]
            _write_npz_atomic(
                self.partial_dir / f"recordings/timestamps/{name}.npz",
                frame_index=np.asarray(sink.frame_index, dtype=np.int64),
                timestamp_monotonic_ns=np.asarray(sink.timestamp_monotonic_ns, dtype=np.int64),
                frame_received_timestamp_ns=np.asarray(sink.frame_received_timestamp_ns, dtype=np.int64),
                device_timestamp_ns=np.asarray(sink.device_timestamp_ns, dtype=np.int64),
            )
        exterior = self._sinks["exterior_image_1_left"]
        _write_npz_atomic(
            self.partial_dir / "recordings/timestamps/exterior_image_2_left.npz",
            frame_index=np.asarray(exterior.frame_index, dtype=np.int64),
            timestamp_monotonic_ns=np.asarray(
                exterior.timestamp_monotonic_ns,
                dtype=np.int64,
            ),
            frame_received_timestamp_ns=np.asarray(
                exterior.frame_received_timestamp_ns,
                dtype=np.int64,
            ),
            device_timestamp_ns=np.asarray(exterior.device_timestamp_ns, dtype=np.int64),
        )
        exterior_1_video = (
            self.partial_dir / "recordings/MP4/exterior_image_1_left.mp4"
        )
        exterior_2_video = (
            self.partial_dir / "recordings/MP4/exterior_image_2_left.mp4"
        )
        try:
            os.link(exterior_1_video, exterior_2_video)
        except OSError:
            shutil.copyfile(exterior_1_video, exterior_2_video)
        gelsight = self._sinks["gelsight_left"]
        gelsight_droid_video = self.partial_dir / "tactile/gelsight_left.mp4"
        gelsight_octopi_video = self.partial_dir / "tactile/gelsight.mp4"
        try:
            os.link(gelsight_droid_video, gelsight_octopi_video)
        except OSError:
            shutil.copyfile(gelsight_droid_video, gelsight_octopi_video)
        gelsight_timestamps = np.asarray(gelsight.timestamp_monotonic_ns, dtype=np.int64)
        _write_npy_atomic(self.partial_dir / "tactile/gelsight_left_timestamps.npy", gelsight_timestamps)
        _write_npz_atomic(
            self.partial_dir / "tactile/gelsight_left_frame_metadata.npz",
            frame_index=np.asarray(gelsight.frame_index, dtype=np.int64),
            timestamp_monotonic_ns=gelsight_timestamps,
            frame_received_timestamp_ns=np.asarray(gelsight.frame_received_timestamp_ns, dtype=np.int64),
            device_timestamp_ns=np.asarray(gelsight.device_timestamp_ns, dtype=np.int64),
            dropped_frame_count=np.asarray(
                [self.camera_streams()["gelsight_left"].dropped_frame_count], dtype=np.int64
            ),
        )
        ati = self.ati_stream()
        ati_samples = (
            ati.samples_between(self.started_ns, self.stopped_ns)
            if ati is not None
            else []
        )
        _write_ati_parquet_atomic(self.partial_dir / "tactile/ati_raw.parquet", ati_samples)
        self._write_octopi_compatibility(gelsight, ati_samples, success, failure_reason)
        atomic_write_json(
            self.partial_dir / "tactile/events.json",
            {
                "schema_version": SCHEMA_VERSION,
                "events": [event.to_dict() for event in sorted(self.events, key=lambda value: value.timestamp_monotonic_ns)],
            },
        )
        atomic_write_json(
            self.partial_dir / "tactile/calibration.json",
            ForceCalibration("unavailable" if not self.config.ati_enabled else "uncalibrated").to_dict(),
        )
        metadata = default_metadata(
            self.config.destination_tray,
            self.config.swatch_uid,
            self.config.split,
            self.config.session_id,
            self.config.episode_id,
        )
        metadata = replace(
            metadata,
            success=success,
            failure_reason=failure_reason,
            operator=self.config.operator,
            robot_id=f"franka@{self.config.robot_ip}",
            camera_serials={
                "exterior_image_1_left": self.config.exterior_serial,
                "exterior_image_2_left": (
                    f"{self.config.exterior_serial}:duplicate_for_openpi"
                ),
                "wrist_image_left": self.config.wrist_serial,
            },
            gelsight_serial=self.config.gelsight_serial,
            ati_serial="not-connected" if not self.config.ati_enabled else "ati-nano17",
            calibration_id="unavailable" if not self.config.ati_enabled else "uncalibrated",
            software_git_commits={"droid": git_commit(Path(__file__).resolve().parents[2])},
        )
        trajectory_present = False
        trajectory_report: dict[str, Any] = {}
        if self.robot_telemetry_path.is_file():
            trajectory_report = write_droid_trajectory_h5(
                self.robot_telemetry_path,
                self.partial_dir / "trajectory.h5",
                success=success,
                failure_reason=failure_reason,
                task_instruction=metadata.task_instruction,
                robot_ip=self.config.robot_ip,
            )
            trajectory_present = True
        elif success:
            raise RuntimeError(
                "real robot telemetry is missing; refusing to mark a sensor-only episode complete"
            )
        metadata = replace(metadata, robot_motion_enabled=trajectory_present)
        atomic_write_json(
            self.partial_dir / f"metadata_{self.config.episode_id}.json",
            {
                **metadata.to_dict(),
                "capture_scope": "droid_trajectory_plus_tactile_sidecar",
                "robot_trajectory_present": trajectory_present,
                "robot_trajectory": trajectory_report,
                "exterior_image_2_left_is_duplicate": True,
            },
        )
        camera_streams = self.camera_streams()
        ati_clock = analyze_timestamps(
            [sample.timestamp_monotonic_ns for sample in ati_samples]
        ).to_dict()
        final_sensor_health = self._final_sensor_health or self.sensor_health_report()
        ati_health = final_sensor_health["ati"]
        report = {
            "schema_version": SCHEMA_VERSION,
            "complete": success,
            "capture_scope": "droid_trajectory_plus_tactile_sidecar",
            "robot_motion_enabled": trajectory_present,
            "robot_trajectory_present": trajectory_present,
            "robot_trajectory": trajectory_report,
            "failure_reason": failure_reason,
            "episode_start_monotonic_ns": self.started_ns,
            "episode_end_monotonic_ns": self.stopped_ns,
            "duration_sec": (self.stopped_ns - self.started_ns) / 1e9,
            "ati_enabled": self.config.ati_enabled,
            "ati_connected": bool(ati_health["ready"]),
            "mandatory_sensor_gate": final_sensor_health,
            "streams": {stream.name: stream.snapshot() for stream in self.streams},
            "video_sinks": {
                name: sink.status() for name, sink in self._sinks.items()
            },
            "clock_reports": {
                name: sink.clock_report() for name, sink in self._sinks.items()
            }
            | {
                "ati_nano17": {
                    **ati_clock,
                    "enabled": self.config.ati_enabled,
                    "connected": bool(ati_health["ready"]),
                }
            },
        }
        atomic_write_json(self.partial_dir / "tactile/capture_report.json", report)
        marker = "COMPLETE.json" if success else "CAPTURE_INCOMPLETE.json"
        atomic_write_json(
            self.partial_dir / marker,
            {
                "complete": success,
                "capture_scope": "droid_trajectory_plus_tactile_sidecar",
                "robot_trajectory_present": trajectory_present,
                "closed_monotonic_ns": self.stopped_ns,
                "failure_reason": failure_reason,
            },
        )

    def _write_octopi_compatibility(
        self,
        gelsight: _StreamingVideoSink,
        ati_samples: list[Any],
        success: bool,
        failure_reason: str,
    ) -> None:
        """Write the Octopi Fabric Session v2 files beside DROID sidecars.

        The MP4 is hard-linked from ``gelsight_left.mp4`` so compatibility does
        not duplicate its disk usage. ATI remains fully preserved at publisher
        rate in both this CSV and the canonical Parquet file.
        """

        tactile_dir = self.partial_dir / "tactile"
        wall_minus_monotonic_ns = time.time_ns() - time.monotonic_ns()
        start_ns = int(self.started_ns or 0)
        frame_rows: list[list[Any]] = []
        width, height = gelsight.size or (self.config.gelsight_width, self.config.gelsight_height)
        for index, monotonic_ns, device_ns in zip(
            gelsight.frame_index,
            gelsight.timestamp_monotonic_ns,
            gelsight.device_timestamp_ns,
        ):
            wall_ns = int(monotonic_ns) + wall_minus_monotonic_ns
            frame_rows.append(
                [
                    index,
                    monotonic_ns,
                    round((int(monotonic_ns) - start_ns) / 1e9, 6),
                    _wall_iso_from_ns(wall_ns),
                    "",
                    True,
                    width,
                    height,
                    "" if int(device_ns) < 0 else int(device_ns),
                    "" if int(device_ns) < 0 else "ns",
                ]
            )
        _write_csv_atomic(
            tactile_dir / "frames_ts.csv",
            [
                "frame_idx",
                "t_monotonic_ns",
                "t_monotonic_sec",
                "t_wall_time_iso",
                "capture_latency_ms",
                "write_success",
                "frame_width",
                "frame_height",
                "camera_timestamp",
                "camera_timestamp_unit",
            ],
            frame_rows,
        )

        valid_fz = np.asarray(
            [
                sample.fz
                for sample in ati_samples[:1000]
                if sample.device_status == "ok" and np.isfinite(sample.fz)
            ],
            dtype=np.float64,
        )
        baseline_fz = float(np.median(valid_fz)) if valid_fz.size else 0.0
        force_rows: list[list[Any]] = []
        estimated_peak_force = 0.0
        for sample in ati_samples:
            valid = sample.device_status == "ok" and np.isfinite(sample.fz)
            normal_force = -(float(sample.fz) - baseline_fz) if valid else ""
            if valid:
                estimated_peak_force = max(estimated_peak_force, float(normal_force))
            force_rows.append(
                [
                    sample.sample_index,
                    sample.timestamp_monotonic_ns,
                    round((sample.timestamp_monotonic_ns - start_ns) / 1e9, 6),
                    _wall_iso_from_ns(sample.timestamp_wall_ns),
                    sample.fx,
                    sample.fy,
                    sample.fz,
                    sample.tx,
                    sample.ty,
                    sample.tz,
                    normal_force,
                    baseline_fz,
                    valid,
                ]
            )
        _write_csv_atomic(
            tactile_dir / "nano17.csv",
            [
                "sample_idx",
                "t_monotonic_ns",
                "t_monotonic_sec",
                "t_wall_time_iso",
                "fx",
                "fy",
                "fz",
                "tx",
                "ty",
                "tz",
                "normal_force_N",
                "baseline_fz",
                "valid",
            ],
            force_rows,
        )

        video_clock = gelsight.clock_report()
        ati_clock = analyze_timestamps(
            [sample.timestamp_monotonic_ns for sample in ati_samples]
        ).to_dict()
        warnings: list[str] = []
        measured_video_fps = float(video_clock.get("measured_hz", 0.0))
        if measured_video_fps and measured_video_fps < 15.0:
            warnings.append(f"low_video_fps:{measured_video_fps:.3f}Hz")
        if not gelsight.background_written:
            warnings.append("missing_background")
        if self.config.ati_enabled and not ati_samples:
            warnings.append("no_force_samples")
        if not success:
            warnings.append(f"incomplete:{failure_reason}")
        session_meta = {
            "schema_version": "fabric_session_v2.1",
            "fabric_id": self.config.swatch_uid,
            "side": "unspecified",
            "session_id": self.config.session_id,
            "episode_id": self.config.episode_id,
            "paths": {
                "label_json": None,
                "fabric_rgb": "../recordings/MP4/exterior_image_1_left.mp4",
                "gelsight_video": "gelsight.mp4",
                "frames_ts": "frames_ts.csv",
                "nano17_csv": "nano17.csv",
                "background": "background.jpg",
            },
            "recording": {
                "session_start_monotonic_ns": self.started_ns,
                "session_end_monotonic_ns": self.stopped_ns,
                "duration_sec": round((int(self.stopped_ns or 0) - start_ns) / 1e9, 6),
                "num_video_frames": gelsight.count,
                "num_force_samples": len(ati_samples),
                "video_fps_configured": self.config.gelsight_fps,
                "video_fps_estimated": measured_video_fps,
                "ati_rate_target_hz": 500.0,
                "ati_rate_measured_hz": float(ati_clock.get("measured_hz", 0.0)),
            },
            "force": {
                "normal_force_mode": "neg_fz",
                "normal_force_calibrated": False,
                "normal_force_usage": "offline Octopi-compatible proxy only; forbidden for closed-loop control",
                "baseline_fz": baseline_fz,
                "target_peak_force_N": None,
                "estimated_peak_force_N": estimated_peak_force,
                "force_sensor": "ATI Nano17",
                "channels": ["fx", "fy", "fz", "tx", "ty", "tz"],
                "units": {
                    "fx": "N",
                    "fy": "N",
                    "fz": "N",
                    "tx": "N_m",
                    "ty": "N_m",
                    "tz": "N_m",
                },
                "endpoint": self.config.ati_endpoint if self.config.ati_enabled else None,
            },
            "sync": {
                "clock": "python_time_monotonic_ns",
                "estimated_delay_sec": None,
                "delay_definition": None,
                "force_time_for_image_frame": "align by t_monotonic_ns; delay not calibrated",
            },
            "rgb": {
                "front_rgb_preserved": True,
                "back_rgb_captured": False,
                "rgb_camera": self.config.exterior_serial,
            },
            "operator_events": [
                {
                    "event": event.name,
                    "t_monotonic_sec": round(
                        (event.timestamp_monotonic_ns - start_ns) / 1e9,
                        6,
                    ),
                }
                for event in sorted(
                    self.events,
                    key=lambda value: value.timestamp_monotonic_ns,
                )
            ],
            "qc": {"bad": not success, "warnings": warnings},
            "hardware": {
                "gelsight_camera": self.config.gelsight_source,
                "gelsight_serial": self.config.gelsight_serial,
                "frame_width": width,
                "frame_height": height,
            },
        }
        atomic_write_json(tactile_dir / "session_meta.json", session_meta)
