#!/usr/bin/env python3
"""Capture D435 RGB and a UVC fisheye concurrently without touching the robot."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.io_utils import atomic_write_json
from fabric_droid.sync.clock import analyze_timestamps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--d435-serial")
    parser.add_argument("--d435-width", type=int, default=640)
    parser.add_argument("--d435-height", type=int, default=480)
    parser.add_argument("--d435-fps", type=int, default=30)
    parser.add_argument("--fisheye-source", default="/dev/video8")
    parser.add_argument("--fisheye-width", type=int, default=1280)
    parser.add_argument("--fisheye-height", type=int, default=720)
    parser.add_argument("--fisheye-fps", type=int, default=30)
    return parser.parse_args()


def _writer(path: Path, width: int, height: int, fps: float) -> tuple[cv2.VideoWriter, Path]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"unable to open video writer: {temporary}")
    return writer, temporary


def _sensor_option(sensor: Any, option: Any) -> float | None:
    return float(sensor.get_option(option)) if sensor.supports(option) else None


def main() -> int:
    args = parse_args()
    if args.duration_sec <= 0:
        raise ValueError("duration-sec must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    context = rs.context()
    devices = list(context.query_devices())
    if not devices:
        raise RuntimeError("RealSense SDK found no device")
    serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
    serial = args.d435_serial or serials[0]
    if serial not in serials:
        raise ValueError(f"D435 serial {serial} not found; connected serials: {serials}")

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        args.d435_width,
        args.d435_height,
        rs.format.bgr8,
        args.d435_fps,
    )
    profile = pipeline.start(config)
    color_sensor = profile.get_device().first_color_sensor()

    source: int | str = int(args.fisheye_source) if args.fisheye_source.isdigit() else args.fisheye_source
    fisheye = cv2.VideoCapture(source, cv2.CAP_V4L2)
    fisheye.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    fisheye.set(cv2.CAP_PROP_FRAME_WIDTH, args.fisheye_width)
    fisheye.set(cv2.CAP_PROP_FRAME_HEIGHT, args.fisheye_height)
    fisheye.set(cv2.CAP_PROP_FPS, args.fisheye_fps)
    if not fisheye.isOpened():
        pipeline.stop()
        raise RuntimeError(f"unable to open fisheye source {source}")

    actual_fisheye = {
        "width": int(fisheye.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(fisheye.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(fisheye.get(cv2.CAP_PROP_FPS)),
        "fourcc": int(fisheye.get(cv2.CAP_PROP_FOURCC)),
        "exposure": float(fisheye.get(cv2.CAP_PROP_EXPOSURE)),
        "white_balance": float(fisheye.get(cv2.CAP_PROP_WB_TEMPERATURE)),
    }
    d435_path = output / "d435_rgb.mp4"
    fisheye_path = output / "fisheye_rgb.mp4"
    d435_writer, d435_temporary = _writer(
        d435_path,
        args.d435_width,
        args.d435_height,
        args.d435_fps,
    )
    fisheye_writer, fisheye_temporary = _writer(
        fisheye_path,
        actual_fisheye["width"],
        actual_fisheye["height"],
        actual_fisheye["fps"],
    )
    d435_host_ns: list[int] = []
    d435_device_ms: list[float] = []
    fisheye_host_ns: list[int] = []
    d435_first = d435_last = None
    fisheye_first = fisheye_last = None
    try:
        for _ in range(15):
            pipeline.wait_for_frames(1000)
            fisheye.grab()
        start_ns = time.monotonic_ns()
        end_ns = start_ns + int(args.duration_sec * 1e9)
        while time.monotonic_ns() < end_ns:
            frames = pipeline.wait_for_frames(1000)
            color = frames.get_color_frame()
            if color:
                image = np.asanyarray(color.get_data())
                now_ns = time.monotonic_ns()
                d435_writer.write(image)
                d435_host_ns.append(now_ns)
                d435_device_ms.append(float(color.get_timestamp()))
                d435_first = image.copy() if d435_first is None else d435_first
                d435_last = image.copy()
            ok, image = fisheye.read()
            now_ns = time.monotonic_ns()
            if ok:
                fisheye_writer.write(image)
                fisheye_host_ns.append(now_ns)
                fisheye_first = image.copy() if fisheye_first is None else fisheye_first
                fisheye_last = image.copy()
    finally:
        d435_writer.release()
        fisheye_writer.release()
        fisheye.release()
        pipeline.stop()
    if not d435_host_ns or not fisheye_host_ns:
        raise RuntimeError(f"empty capture: D435 frames={len(d435_host_ns)}, fisheye frames={len(fisheye_host_ns)}")
    os.replace(d435_temporary, d435_path)
    os.replace(fisheye_temporary, fisheye_path)
    cv2.imwrite(str(output / "d435_first.jpg"), d435_first)
    cv2.imwrite(str(output / "d435_last.jpg"), d435_last)
    cv2.imwrite(str(output / "fisheye_first.jpg"), fisheye_first)
    cv2.imwrite(str(output / "fisheye_last.jpg"), fisheye_last)
    with (output / "timestamps.npz").open("wb") as stream:
        np.savez(
            stream,
            d435_host_monotonic_ns=np.asarray(d435_host_ns, dtype=np.int64),
            d435_device_timestamp_ms=np.asarray(d435_device_ms, dtype=np.float64),
            fisheye_host_monotonic_ns=np.asarray(fisheye_host_ns, dtype=np.int64),
        )
    d435_clock = analyze_timestamps(d435_host_ns)
    fisheye_clock = analyze_timestamps(fisheye_host_ns)
    report = {
        "pass": (d435_clock.measured_hz >= args.d435_fps * 0.8 and fisheye_clock.measured_hz >= args.fisheye_fps * 0.8),
        "robot_motion_enabled": False,
        "d435": {
            "serial": serial,
            "name": profile.get_device().get_info(rs.camera_info.name),
            "firmware": profile.get_device().get_info(rs.camera_info.firmware_version),
            "usb": profile.get_device().get_info(rs.camera_info.usb_type_descriptor),
            "width": args.d435_width,
            "height": args.d435_height,
            "requested_fps": args.d435_fps,
            "exposure": _sensor_option(color_sensor, rs.option.exposure),
            "white_balance": _sensor_option(color_sensor, rs.option.white_balance),
            "clock": d435_clock.to_dict(),
            "first_last_mean_abs_delta": float(np.mean(cv2.absdiff(d435_first, d435_last))),
            "video": str(d435_path),
        },
        "fisheye": {
            "source": str(source),
            "requested": {
                "width": args.fisheye_width,
                "height": args.fisheye_height,
                "fps": args.fisheye_fps,
                "fourcc": "MJPG",
            },
            "actual": actual_fisheye,
            "clock": fisheye_clock.to_dict(),
            "first_last_mean_abs_delta": float(np.mean(cv2.absdiff(fisheye_first, fisheye_last))),
            "video": str(fisheye_path),
        },
    }
    atomic_write_json(output / "capture_report.json", report)
    print(report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
