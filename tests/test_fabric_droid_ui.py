from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
import inspect
from pathlib import Path

import numpy as np
import pytest

from fabric_droid.sensors.camera import (
    CameraSpec,
    CameraStream,
    _lock_uvc_exposure_v4l2,
    _lock_uvc_white_balance_v4l2,
    _scaled_exposure_absolute,
)
from fabric_droid.sensors.ati import ATISample, ATIStream
from fabric_droid.ui.devices import (
    CameraDevice,
    discover_devices,
    is_gelsight_device,
    preferred_gelsight_serial,
    preferred_wrist_uvc_serial,
)
from fabric_droid.ui.episodes import discover_episodes
from fabric_droid.ui.session import (
    CollectionConfig,
    StreamedSensorEpisode,
    _StreamingVideoSink,
)


def test_camera_preview_buffer_is_bounded_and_indices_remain_global() -> None:
    callbacks: list[int] = []
    stream = CameraStream(
        CameraSpec("test", "uvc", "unused", "serial", 8, 6, 30),
        max_buffer_frames=2,
        frame_callback=lambda frame: callbacks.append(frame.frame_index),
    )
    for value in range(5):
        stream._append(np.full((6, 8, 3), value, dtype=np.uint8))  # noqa: SLF001
    assert [frame.frame_index for frame in stream.frames] == [3, 4]
    assert callbacks == [0, 1, 2, 3, 4]
    assert stream.snapshot()["count"] == 5
    assert stream.snapshot()["buffered_count"] == 2
    assert stream.latest_frame() is not None
    assert stream.latest_frame().frame_index == 4


def test_uvc_manual_exposure_scale_is_linear_clamped_and_step_aligned() -> None:
    assert _scaled_exposure_absolute(100, 1.15, 1, 1000, 1) == 115
    assert _scaled_exposure_absolute(100, 1.15, 10, 1000, 5) == 115
    assert _scaled_exposure_absolute(990, 1.15, 1, 1000, 1) == 1000


def test_uvc_exposure_supports_fisheye_control_aliases(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        argument = command[-1]
        if argument == "--list-ctrls":
            output = "\n".join(
                (
                    "Camera Controls",
                    " auto_exposure 0x009a0901 (menu) : min=0 max=3 default=3 value=1",
                    " exposure_time_absolute 0x009a0902 (int) : "
                    "min=1 max=10000 step=1 default=166 value=333",
                )
            )
        elif argument == "--get-ctrl=exposure_time_absolute":
            output = "exposure_time_absolute: 333\n"
        elif argument == "--get-ctrl=auto_exposure,exposure_time_absolute":
            output = "auto_exposure: 1\nexposure_time_absolute: 383\n"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("fabric_droid.sensors.camera.subprocess.run", fake_run)
    report = _lock_uvc_exposure_v4l2(
        "/dev/video10",
        exposure_absolute=383,
    )
    assert report["auto_exposure"] is False
    assert report["exposure_initial"] == 333
    assert report["exposure"] == 383
    assert report["exposure_control"] == "v4l2 exposure_time_absolute"
    assert any(
        command[-1] == "--set-ctrl=auto_exposure=1"
        for command in commands
    )
    assert any(
        command[-1] == "--set-ctrl=exposure_time_absolute=383"
        for command in commands
    )


def test_uvc_white_balance_locks_fisheye_controls(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(command)
        argument = command[-1]
        if argument == "--list-ctrls":
            output = "\n".join(
                (
                    "User Controls",
                    " white_balance_automatic 0x0098090c (bool) : "
                    "default=1 value=1",
                    " white_balance_temperature 0x0098091a (int) : "
                    "min=2800 max=6500 step=10 default=4600 value=4600 "
                    "flags=inactive",
                )
            )
        elif (
            argument
            == "--get-ctrl=white_balance_automatic,white_balance_temperature"
        ):
            output = (
                "white_balance_automatic: 0\n"
                "white_balance_temperature: 4200\n"
            )
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("fabric_droid.sensors.camera.subprocess.run", fake_run)
    report = _lock_uvc_white_balance_v4l2("/dev/video10", 4200)
    assert report["auto_white_balance"] is False
    assert report["white_balance"] == 4200
    assert any(
        command[-1] == "--set-ctrl=white_balance_automatic=0"
        for command in commands
    )
    assert any(
        command[-1] == "--set-ctrl=white_balance_temperature=4200"
        for command in commands
    )


def test_camera_readiness_requires_fresh_frames(monkeypatch) -> None:
    stream = CameraStream(
        CameraSpec("test", "uvc", "unused", "serial", 8, 6, 30),
        max_buffer_frames=2,
    )
    assert not stream.readiness()["ready"]
    stream._append(np.zeros((6, 8, 3), dtype=np.uint8))  # noqa: SLF001
    stream._append(np.ones((6, 8, 3), dtype=np.uint8))  # noqa: SLF001
    assert stream.readiness()["ready"]
    latest_ns = stream.latest_frame().timestamp_monotonic_ns
    assert not stream.readiness(since_monotonic_ns=latest_ns + 1)["ready"]
    with stream._lock:  # noqa: SLF001
        stream._max_inter_frame_gap_ns = 1_100_000_000  # noqa: SLF001
    gap_report = stream.readiness(max_gap_sec=1.0)
    assert not gap_report["ready"]
    assert any(
        "historical frame gap" in reason
        for reason in gap_report["reasons"]
    )
    monkeypatch.setattr(
        "fabric_droid.sensors.camera.time.monotonic_ns",
        lambda: latest_ns + 2_000_000_000,
    )
    report = stream.readiness(max_age_sec=1.0)
    assert not report["ready"]
    assert any("stale" in reason for reason in report["reasons"])


def test_camera_gap_monitor_counts_start_to_first_episode_frame(
    monkeypatch,
) -> None:
    stream = CameraStream(
        CameraSpec("test", "uvc", "unused", "serial", 8, 6, 30),
        max_buffer_frames=2,
    )
    times = iter((9_999_000_000, 11_100_000_000, 11_100_000_000))
    monkeypatch.setattr(
        "fabric_droid.sensors.camera.time.monotonic_ns",
        lambda: next(times),
    )
    stream._append(np.ones((6, 8, 3), dtype=np.uint8))  # noqa: SLF001
    stream.reset_gap_monitor(10_000_000_000)
    stream._append(np.ones((6, 8, 3), dtype=np.uint8))  # noqa: SLF001
    report = stream.readiness(max_gap_sec=1.0)
    assert report["max_gap_sec_observed"] == pytest.approx(1.1)
    assert any("historical frame gap" in reason for reason in report["reasons"])


def test_camera_preview_can_downsize_before_buffering() -> None:
    stream = CameraStream(
        CameraSpec("test", "uvc", "unused", "serial", 64, 48, 30),
        max_buffer_frames=2,
        frame_output_size=(16, 12),
    )
    stream._append(np.zeros((48, 64, 3), dtype=np.uint8))  # noqa: SLF001
    assert stream.latest_frame().image_bgr.shape == (12, 16, 3)
    assert stream.snapshot()["frame_output_size"] == (16, 12)


def test_camera_snapshot_reports_rate_since_monitor_reset(monkeypatch) -> None:
    stream = CameraStream(
        CameraSpec("test", "uvc", "unused", "serial", 8, 6, 30),
        max_buffer_frames=2,
    )
    times = iter(
        (
            900_000_000,
            1_100_000_000,
            1_200_000_000,
            2_000_000_000,
        )
    )
    monkeypatch.setattr(
        "fabric_droid.sensors.camera.time.monotonic_ns",
        lambda: next(times),
    )
    stream._append(np.zeros((6, 8, 3), dtype=np.uint8))  # noqa: SLF001
    stream.reset_gap_monitor(1_000_000_000)
    stream._append(np.zeros((6, 8, 3), dtype=np.uint8))  # noqa: SLF001
    stream._append(np.zeros((6, 8, 3), dtype=np.uint8))  # noqa: SLF001
    snapshot = stream.snapshot()
    assert snapshot["rate_monitor_elapsed_sec"] == pytest.approx(1.0)
    assert snapshot["measured_hz"] == pytest.approx(2.0)


def test_camera_request_stop_only_signals_worker() -> None:
    stream = CameraStream(
        CameraSpec("test", "uvc", "unused", "serial", 8, 6, 30),
    )
    assert not stream._stop_event.is_set()  # noqa: SLF001
    stream.request_stop()
    assert stream._stop_event.is_set()  # noqa: SLF001
    stream.stop(timeout=0.01)


def test_d435_tint_adjusts_green_red_and_leaves_blue_unchanged() -> None:
    stream = CameraStream(
        CameraSpec("test", "d435", "unused", "serial", 8, 6, 30),
        max_buffer_frames=2,
    )
    image = np.full((6, 8, 3), 100, dtype=np.uint8)
    assert stream.set_tint(100) is True
    stream._append(image)  # noqa: SLF001
    positive = stream.latest_frame().image_bgr[0, 0].tolist()
    assert positive == [100, 75, 125]

    assert stream.set_tint(-100) is True
    stream._append(image)  # noqa: SLF001
    negative = stream.latest_frame().image_bgr[0, 0].tolist()
    assert negative == [100, 125, 75]


def test_device_discovery_deduplicates_realsense_and_prefers_stable_uvc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "video2").touch()
    (tmp_path / "video14").touch()
    by_id = tmp_path / "v4l" / "by-id"
    by_id.mkdir(parents=True)
    (by_id / "usb-GelSight_ABC-video-index0").symlink_to(tmp_path / "video14")

    properties = {
        "video2": "\n".join(
            (
                "ID_V4L_PRODUCT=Intel(R) RealSense(TM) Depth Camera 435",
                "ID_SERIAL_SHORT=123",
                "ID_V4L_CAPABILITIES=:capture:",
            )
        ),
        "video14": "\n".join(
            (
                "ID_V4L_PRODUCT=GelSight Mini",
                "ID_SERIAL_SHORT=GS1",
                "ID_V4L_CAPABILITIES=:capture:",
            )
        ),
    }

    def fake_run(command, **kwargs):
        del kwargs
        name = Path(command[-1].split("=", 1)[-1]).name
        return subprocess.CompletedProcess(command, 0, properties.get(name, ""), "")

    monkeypatch.setattr("fabric_droid.ui.devices.subprocess.run", fake_run)
    monkeypatch.setattr(
        "fabric_droid.ui.devices._discover_realsense_python",
        lambda: [],
    )
    inventory = discover_devices(tmp_path)
    assert [device.serial for device in inventory.realsense] == ["123"]
    assert [device.serial for device in inventory.uvc] == ["GS1"]
    assert inventory.uvc[0].source.endswith("usb-GelSight_ABC-video-index0")


def test_gelsight_selection_overrides_stale_laptop_webcam_profile() -> None:
    laptop = CameraDevice(
        "uvc",
        "200901010001",
        "HD Webcam",
        "/dev/v4l/by-id/usb-Chicony_HD_Webcam-video-index0",
    )
    gelsight = CameraDevice(
        "uvc",
        "2DWF0RJM",
        "USB Camera",
        "/dev/v4l/by-id/usb-Arducam_GelSight_Mini_2DWF0RJM-video-index0",
    )
    assert not is_gelsight_device(laptop)
    assert is_gelsight_device(gelsight)
    assert (
        preferred_gelsight_serial([laptop, gelsight], laptop.serial)
        == gelsight.serial
    )
    assert (
        preferred_gelsight_serial([laptop, gelsight], gelsight.serial)
        == gelsight.serial
    )


def test_gelsight_selection_keeps_manual_uvc_when_no_gelsight_is_present() -> None:
    laptop = CameraDevice("uvc", "webcam", "HD Webcam", "/dev/video0")
    assert preferred_gelsight_serial([laptop], laptop.serial) == laptop.serial


def test_wrist_selection_prefers_external_uvc_over_laptop_webcam() -> None:
    laptop = CameraDevice(
        "uvc",
        "200901010001",
        "HD Webcam",
        "/dev/video0",
        "pci-usb-0:5:1.0",
    )
    fisheye = CameraDevice(
        "uvc",
        "200901010001@9",
        "USB camera",
        "/dev/video10",
        "pci-usb-0:9:1.0",
    )
    gelsight = CameraDevice(
        "uvc",
        "2DWF0RJM",
        "GelSight Mini",
        "/dev/video12",
    )
    assert (
        preferred_wrist_uvc_serial([laptop, fisheye, gelsight], None)
        == fisheye.serial
    )
    assert (
        preferred_wrist_uvc_serial(
            [laptop, fisheye, gelsight],
            laptop.serial,
        )
        == laptop.serial
    )


def test_device_discovery_preserves_uvc_cameras_with_duplicate_serials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "video0").touch()
    (tmp_path / "video10").touch()
    properties = {
        "video0": "\n".join(
            (
                "ID_V4L_PRODUCT=HD Webcam",
                "ID_SERIAL_SHORT=200901010001",
                "ID_V4L_CAPABILITIES=:capture:",
                "ID_PATH=pci-0000:00:14.0-usb-0:5:1.0",
            )
        ),
        "video10": "\n".join(
            (
                "ID_V4L_PRODUCT=USB camera",
                "ID_SERIAL_SHORT=200901010001",
                "ID_V4L_CAPABILITIES=:capture:",
                "ID_PATH=pci-0000:00:14.0-usb-0:9:1.0",
            )
        ),
    }

    def fake_run(command, **kwargs):
        del kwargs
        name = Path(command[-1].split("=", 1)[-1]).name
        return subprocess.CompletedProcess(command, 0, properties[name], "")

    monkeypatch.setattr("fabric_droid.ui.devices.subprocess.run", fake_run)
    monkeypatch.setattr(
        "fabric_droid.ui.devices._discover_realsense_python",
        lambda: [],
    )
    inventory = discover_devices(tmp_path)
    assert [device.serial for device in inventory.uvc] == [
        "200901010001",
        "200901010001@9",
    ]
    assert [device.source for device in inventory.uvc] == [
        str(tmp_path / "video0"),
        str(tmp_path / "video10"),
    ]


def test_device_discovery_prefers_sdk_serial_for_same_physical_port(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "video2").touch()

    def fake_run(command, **kwargs):
        del kwargs
        output = "\n".join(
            (
                "ID_V4L_PRODUCT=Intel(R) RealSense(TM) Depth Camera 435",
                "ID_SERIAL_SHORT=V4L_SERIAL",
                "ID_V4L_CAPABILITIES=:capture:",
                "ID_PATH=pci-0000:00:14.0-usb-0:1.1:1.0",
            )
        )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("fabric_droid.ui.devices.subprocess.run", fake_run)
    monkeypatch.setattr(
        "fabric_droid.ui.devices._discover_realsense_python",
        lambda: [
            CameraDevice(
                "d435",
                "SDK_SERIAL",
                "Intel RealSense D435",
                "SDK_SERIAL",
                "/sys/devices/usb2/2-1/2-1.1/2-1.1:1.0/video4linux/video2",
            )
        ],
    )
    inventory = discover_devices(tmp_path)
    assert [device.serial for device in inventory.realsense] == ["SDK_SERIAL"]


def test_streaming_video_sink_writes_without_retaining_images(tmp_path: Path) -> None:
    from fabric_droid.sensors.camera import CameraFrame

    sink = _StreamingVideoSink(tmp_path / "test.mp4", 10)
    for index in range(4):
        timestamp = time.monotonic_ns()
        sink.append(
            CameraFrame(
                index,
                timestamp,
                timestamp,
                index,
                np.full((24, 32, 3), index * 20, dtype=np.uint8),
            )
        )
    sink.close()
    assert sink.count == 4
    assert (tmp_path / "test.mp4").stat().st_size > 0
    status = sink.status()
    assert status["queued_frame_count"] == 4
    assert status["written_frame_count"] == 4
    assert status["pending_queue_frames"] == 0
    assert status["writer_failed"] is False
    assert not any(isinstance(value, np.ndarray) for value in vars(sink).values())


def test_streaming_gelsight_sink_writes_octopi_background(tmp_path: Path) -> None:
    from fabric_droid.sensors.camera import CameraFrame

    background = tmp_path / "background.jpg"
    sink = _StreamingVideoSink(
        tmp_path / "gelsight.mp4",
        25,
        background_path=background,
        output_size=(32, 24),
    )
    timestamp = time.monotonic_ns()
    sink.append(
        CameraFrame(
            0,
            timestamp,
            timestamp,
            -1,
            np.full((48, 64, 3), 127, dtype=np.uint8),
        )
    )
    sink.close()
    assert background.stat().st_size > 0
    assert sink.background_written is True
    import cv2

    assert cv2.imread(str(background)).shape[:2] == (24, 32)
    capture = cv2.VideoCapture(str(tmp_path / "gelsight.mp4"))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 32
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 24
    capture.release()


def test_gelsight_video_is_reencoded_at_measured_fps(tmp_path: Path) -> None:
    import cv2
    from fabric_droid.sensors.camera import CameraFrame

    sink = _StreamingVideoSink(
        tmp_path / "gelsight.mp4",
        25,
        output_size=(32, 24),
        reencode_measured_fps=True,
    )
    start_ns = 1_000_000_000
    for index in range(6):
        timestamp = start_ns + index * 100_000_000
        sink.append(
            CameraFrame(
                index,
                timestamp,
                timestamp,
                -1,
                np.full((48, 64, 3), index * 20, dtype=np.uint8),
            )
        )
    # _append normally supplies these host timestamps; use deterministic ones
    # here so the writer must correct 25 fps to 10 fps.
    sink.timestamp_monotonic_ns = [
        start_ns + index * 100_000_000 for index in range(6)
    ]
    sink.close()
    capture = cv2.VideoCapture(str(sink.path))
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.1)
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    capture.release()
    assert sink.encoded_fps == pytest.approx(10.0)


def test_octopi_compatibility_sidecars_preserve_all_force_samples(tmp_path: Path) -> None:
    config = CollectionConfig(
        output_root=tmp_path,
        episode_id="episode_001",
        session_id="session_001",
        swatch_uid="F001",
        destination_tray="target_tray",
        split="train",
        operator="tester",
        wrist_serial="wrist",
        exterior_serial="exterior",
        gelsight_source="/dev/video8",
        gelsight_serial="GS1",
        ati_enabled=True,
    )
    episode = StreamedSensorEpisode(config)
    episode.partial_dir.mkdir()
    (episode.partial_dir / "tactile").mkdir()
    episode.started_ns = 1_000_000_000
    episode.stopped_ns = 2_000_000_000
    sink = _StreamingVideoSink(episode.partial_dir / "tactile/gelsight_left.mp4", 25)
    sink.frame_index = [0, 1]
    sink.timestamp_monotonic_ns = [1_100_000_000, 1_150_000_000]
    sink.frame_received_timestamp_ns = list(sink.timestamp_monotonic_ns)
    sink.device_timestamp_ns = [-1, 123]
    sink.size = (640, 480)
    sink.background_written = True
    samples = [
        ATISample(
            1_100_000_000 + index * 2_000_000,
            10_000_000_000 + index * 2_000_000,
            index,
            1.0,
            2.0,
            -float(index),
            0.1,
            0.2,
            0.3,
        )
        for index in range(3)
    ]

    episode._write_octopi_compatibility(sink, samples, True, "")  # noqa: SLF001

    tactile = episode.partial_dir / "tactile"
    with (tactile / "frames_ts.csv").open(newline="", encoding="utf-8") as stream:
        frame_rows = list(csv.DictReader(stream))
    with (tactile / "nano17.csv").open(newline="", encoding="utf-8") as stream:
        force_rows = list(csv.DictReader(stream))
    meta = json.loads((tactile / "session_meta.json").read_text(encoding="utf-8"))
    assert len(frame_rows) == 2
    assert frame_rows[0]["frame_width"] == "640"
    assert frame_rows[0]["frame_height"] == "480"
    assert len(force_rows) == len(samples)
    assert [int(row["sample_idx"]) for row in force_rows] == [0, 1, 2]
    assert meta["schema_version"] == "fabric_session_v2.1"
    assert meta["recording"]["num_video_frames"] == 2
    assert meta["recording"]["num_force_samples"] == 3
    assert meta["force"]["normal_force_calibrated"] is False
    assert meta["paths"]["gelsight_video"] == "gelsight.mp4"


def test_episode_sensor_health_requires_every_camera_and_nonzero_ati(
    tmp_path: Path,
) -> None:
    config = CollectionConfig(
        output_root=tmp_path,
        episode_id="episode_gate",
        session_id="session_gate",
        swatch_uid="F001",
        destination_tray="target_tray",
        split="train",
        operator="tester",
        wrist_serial="wrist",
        exterior_serial="exterior",
        gelsight_source="/dev/video8",
        gelsight_serial="GS1",
    )
    episode = StreamedSensorEpisode(config)
    camera_streams: list[CameraStream] = []
    for spec in episode._camera_specs():  # noqa: SLF001
        stream = CameraStream(spec, max_buffer_frames=2)
        stream._append(np.zeros((24, 32, 3), dtype=np.uint8))  # noqa: SLF001
        stream._append(np.ones((24, 32, 3), dtype=np.uint8))  # noqa: SLF001
        camera_streams.append(stream)
    ati = ATIStream()
    now = time.monotonic_ns()
    with ati._lock:  # noqa: SLF001
        ati.samples = [
            ATISample(
                now - (63 - index) * 2_000_000,
                time.time_ns(),
                index,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
            )
            for index in range(64)
        ]
        ati._total_sample_count = 64  # noqa: SLF001
    episode.streams = [*camera_streams, ati]
    assert episode.sensor_health_report()["ready"]
    episode.streams = [*camera_streams[:-1], ati]
    report = episode.sensor_health_report()
    assert not report["ready"]
    assert any("gelsight_left: stream is missing" in reason for reason in report["reasons"])


def test_collection_uses_uvc_fisheye_for_wrist_and_d435_for_exterior(
    tmp_path: Path,
) -> None:
    config = CollectionConfig(
        output_root=tmp_path,
        episode_id="episode_fisheye",
        session_id="session_fisheye",
        swatch_uid="F001",
        destination_tray="target_tray",
        split="train",
        operator="tester",
        wrist_serial="fisheye@9",
        wrist_kind="uvc",
        wrist_source="/dev/v4l/by-path/wrist-fisheye",
        wrist_pixel_format="MJPG",
        wrist_exposure_absolute=460,
        wrist_white_balance_temperature=4200,
        exterior_serial="d435-exterior",
        gelsight_source="/dev/v4l/by-id/gelsight",
        gelsight_serial="2DWF0RJM",
    )
    specs = {
        spec.name: spec
        for spec in StreamedSensorEpisode(config)._camera_specs()  # noqa: SLF001
    }
    wrist = specs["wrist_image_left"]
    exterior = specs["exterior_image_1_left"]
    assert wrist.kind == "uvc"
    assert wrist.source == "/dev/v4l/by-path/wrist-fisheye"
    assert wrist.pixel_format == "MJPG"
    assert wrist.uvc_exposure_scale is None
    assert wrist.uvc_exposure_absolute == 460
    assert wrist.uvc_white_balance_temperature == 4200
    assert (wrist.width, wrist.height, wrist.fps) == (640, 480, 30)
    assert exterior.kind == "d435"
    assert exterior.source == "d435-exterior"
    assert (exterior.width, exterior.height, exterior.fps) == (640, 480, 30)


def test_collection_defaults_to_d435_wrist_and_d435_exterior(
    tmp_path: Path,
) -> None:
    config = CollectionConfig(
        output_root=tmp_path,
        episode_id="episode_dual_d435",
        session_id="session_dual_d435",
        swatch_uid="F001",
        destination_tray="target_tray",
        split="train",
        operator="tester",
        wrist_serial="d435-wrist",
        exterior_serial="d435-exterior",
        gelsight_source="/dev/v4l/by-id/gelsight",
        gelsight_serial="2DWF0RJM",
    )
    specs = {
        spec.name: spec
        for spec in StreamedSensorEpisode(config)._camera_specs()  # noqa: SLF001
    }
    wrist = specs["wrist_image_left"]
    exterior = specs["exterior_image_1_left"]
    assert wrist.kind == "d435"
    assert wrist.source == "d435-wrist"
    assert wrist.require_usb3 is True
    assert (wrist.width, wrist.height, wrist.fps) == (640, 480, 30)
    assert wrist.exposure == pytest.approx(141)
    assert wrist.white_balance == pytest.approx(3780)
    assert wrist.tint == pytest.approx(-17)
    assert exterior.kind == "d435"
    assert exterior.require_usb3 is True
    assert exterior.source == "d435-exterior"
    assert (exterior.width, exterior.height, exterior.fps) == (640, 480, 30)
    assert exterior.exposure == pytest.approx(141)
    assert exterior.white_balance == pytest.approx(3780)
    assert exterior.tint == pytest.approx(-17)


def test_episode_start_rejects_live_but_all_zero_ati(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FastCamera(CameraStream):
        def run(self) -> None:
            while not self._stop_event.is_set():  # noqa: SLF001
                self._append(np.ones((24, 32, 3), dtype=np.uint8))  # noqa: SLF001
                time.sleep(0.005)

    class ZeroATI(ATIStream):
        def run(self) -> None:
            while not self._stop_event.is_set():  # noqa: SLF001
                timestamp = time.monotonic_ns()
                with self._lock:  # noqa: SLF001
                    index = self._total_sample_count  # noqa: SLF001
                    self.samples.append(
                        ATISample(
                            timestamp,
                            time.time_ns(),
                            index,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        )
                    )
                    self._total_sample_count += 1  # noqa: SLF001
                time.sleep(0.001)

    monkeypatch.setattr("fabric_droid.ui.session.CameraStream", FastCamera)
    monkeypatch.setattr("fabric_droid.ui.session.ATIStream", ZeroATI)
    config = CollectionConfig(
        output_root=tmp_path,
        episode_id="episode_zero_ati",
        session_id="session_gate",
        swatch_uid="F001",
        destination_tray="target_tray",
        split="train",
        operator="tester",
        wrist_serial="wrist",
        exterior_serial="exterior",
        gelsight_source="/dev/video8",
        gelsight_serial="GS1",
        gelsight_width=32,
        gelsight_height=24,
    )
    episode = StreamedSensorEpisode(config)
    with pytest.raises(RuntimeError, match="all zero"):
        episode.start(warmup_timeout_sec=0.2)
    marker = json.loads(
        (episode.partial_dir / "CAPTURE_INCOMPLETE.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["phase"] == "warmup"
    assert "all zero" in marker["error"]
    assert not any(stream._thread is not None for stream in episode.streams)  # noqa: SLF001


def test_episode_start_requires_post_start_samples_and_video_sink_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FastCamera(CameraStream):
        def run(self) -> None:
            while not self._stop_event.is_set():  # noqa: SLF001
                self._append(np.ones((24, 32, 3), dtype=np.uint8))  # noqa: SLF001
                time.sleep(1.0 / self.spec.fps)

    class LiveATI(ATIStream):
        def run(self) -> None:
            while not self._stop_event.is_set():  # noqa: SLF001
                timestamp = time.monotonic_ns()
                with self._lock:  # noqa: SLF001
                    index = self._total_sample_count  # noqa: SLF001
                    self.samples.append(
                        ATISample(
                            timestamp,
                            time.time_ns(),
                            index,
                            0.0,
                            0.0,
                            0.2,
                            0.0,
                            0.0,
                            0.0,
                        )
                    )
                    self._total_sample_count += 1  # noqa: SLF001
                time.sleep(0.001)

    monkeypatch.setattr("fabric_droid.ui.session.CameraStream", FastCamera)
    monkeypatch.setattr("fabric_droid.ui.session.ATIStream", LiveATI)
    config = CollectionConfig(
        output_root=tmp_path,
        episode_id="episode_live_gate",
        session_id="session_gate",
        swatch_uid="F001",
        destination_tray="target_tray",
        split="train",
        operator="tester",
        wrist_serial="wrist",
        exterior_serial="exterior",
        gelsight_source="/dev/video8",
        gelsight_serial="GS1",
        gelsight_width=32,
        gelsight_height=24,
    )
    episode = StreamedSensorEpisode(config)
    episode.start(warmup_timeout_sec=0.5)
    report = episode.sensor_health_report()
    assert report["ready"], report["reasons"]
    assert report["since_monotonic_ns"] == episode.started_ns
    assert all(
        sink["queued_frame_count"] >= 2
        for sink in report["video_sinks"].values()
    )
    persisted: list[tuple[bool, str]] = []
    monkeypatch.setattr(
        episode,
        "_persist",
        lambda success, reason: persisted.append((success, reason)),
    )
    camera = next(iter(episode.camera_streams().values()))
    with camera._lock:  # noqa: SLF001
        camera._max_inter_frame_gap_ns = 1_100_000_000  # noqa: SLF001
    assert episode.stop(success=True) == episode.partial_dir
    assert persisted[0][0] is False
    assert "historical frame gap" in persisted[0][1]


def test_callback_and_poll_can_run_concurrently() -> None:
    stream = CameraStream(
        CameraSpec("test", "uvc", "unused", "serial", 8, 6, 30),
        max_buffer_frames=2,
    )

    def append() -> None:
        for value in range(100):
            stream._append(np.full((6, 8, 3), value, dtype=np.uint8))  # noqa: SLF001

    thread = threading.Thread(target=append)
    thread.start()
    while thread.is_alive():
        stream.latest_frame(copy_image=True)
        stream.snapshot()
    thread.join()
    assert stream.snapshot()["count"] == 100


def test_episode_listing_reports_complete_and_incomplete(tmp_path: Path) -> None:
    complete = tmp_path / "episode_complete"
    (complete / "tactile").mkdir(parents=True)
    (complete / "metadata_episode_complete.json").write_text(
        '{"episode_id":"episode_complete","swatch_uid":"cloth_1","split":"train"}',
        encoding="utf-8",
    )
    (complete / "SENSOR_CAPTURE_COMPLETE.json").write_text('{"complete":true}', encoding="utf-8")
    (complete / "tactile/capture_report.json").write_text(
        """
        {
          "complete": true,
          "duration_sec": 12.5,
          "clock_reports": {
            "exterior_image_1_left": {"count": 360},
            "wrist_image_left": {"count": 359},
            "gelsight_left": {"count": 300},
            "ati_nano17": {"count": 6240, "measured_hz": 499.5}
          }
        }
        """,
        encoding="utf-8",
    )
    incomplete = tmp_path / ".episode_partial.inprogress"
    incomplete.mkdir()
    (incomplete / "CAPTURE_INCOMPLETE.json").write_text('{"complete":false}', encoding="utf-8")

    episodes = discover_episodes(tmp_path)
    by_id = {episode.episode_id: episode for episode in episodes}
    assert by_id["episode_complete"].status == "COMPLETE"
    assert by_id["episode_complete"].camera_counts == (360, 359, 300)
    assert by_id["episode_complete"].ati_count == 6240
    assert by_id["episode_complete"].ati_hz == 499.5
    assert by_id["episode_partial"].status == "INCOMPLETE"


def test_episode_listing_skips_unreadable_non_episode_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "lost+found").mkdir()
    original_is_file = Path.is_file

    def guarded_is_file(path: Path) -> bool:
        if "lost+found" in path.parts:
            raise PermissionError(path)
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    assert discover_episodes(tmp_path) == []


def test_robot_button_workflow_has_no_input_dialogs() -> None:
    from fabric_droid.ui.app import CollectionWindow

    source = inspect.getsource(CollectionWindow)
    assert "QInputDialog" not in source
    assert '"--non-interactive-ui"' in source
    assert 'process.write(b"SAVE_HOME\\n")' in source
    assert 'process.write(b"ENABLE FREEDRIVE\\n")' not in source
    assert 'process.write(b"MOVE TO HOME\\n")' in source
    assert "_pending_recording_config" in source
    assert "franka_quest_teleop.py" in source
    assert '"--enable-gripper"' in source
    assert '"--gripper-max-closedness"' in source
    assert '"0.98"' in source
    assert '"--telemetry-output"' in source
    assert "self.white_balance_slider" in source
    assert "self.tint_slider" in source
    assert "rgb_tint=self.current_d435_tint()" in source
    assert "robot_telemetry_path" in source
    assert source.index('"--rate"') < source.index('"--telemetry-output"')
    assert "_pending_recording_stop_success" in source
    assert "franka_gripper_control.py" in source
    assert '"--open"' in source
    assert source.index("def _gripper_open_finished") < source.index(
        "def _start_go_home_motion"
    )
