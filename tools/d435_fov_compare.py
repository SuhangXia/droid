#!/usr/bin/env python3
"""Capture and visualize the D435 RGB field of view for 4:3 and 16:9 modes."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Capture:
    image: np.ndarray
    width: int
    height: int
    hfov_deg: float
    vfov_deg: float
    fx: float
    ppx: float


def _field_of_view(width: int, height: int, fx: float, fy: float) -> tuple[float, float]:
    hfov = math.degrees(2.0 * math.atan(float(width) / (2.0 * float(fx))))
    vfov = math.degrees(2.0 * math.atan(float(height) / (2.0 * float(fy))))
    return hfov, vfov


def _set_manual_color_controls(sensor: Any, rs: Any, exposure: float, white_balance: float) -> None:
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 0.0)
    if sensor.supports(rs.option.exposure):
        exposure_range = sensor.get_option_range(rs.option.exposure)
        if exposure_range.min <= exposure <= exposure_range.max:
            sensor.set_option(rs.option.exposure, exposure)

    if sensor.supports(rs.option.enable_auto_white_balance):
        sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
    if sensor.supports(rs.option.white_balance):
        white_balance_range = sensor.get_option_range(rs.option.white_balance)
        if white_balance_range.min <= white_balance <= white_balance_range.max:
            sensor.set_option(rs.option.white_balance, white_balance)


def _capture_mode(
    rs: Any,
    serial: str,
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    exposure: float,
    white_balance: float,
) -> Capture:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        profile = pipeline.start(config)
        sensor = profile.get_device().first_color_sensor()
        _set_manual_color_controls(sensor, rs, exposure, white_balance)

        frame = None
        for _ in range(max(1, warmup_frames)):
            frames = pipeline.wait_for_frames(5000)
            candidate = frames.get_color_frame()
            if candidate:
                frame = candidate
        if frame is None:
            raise RuntimeError(f"no color frame received at {width}x{height}@{fps}")

        video_profile = frame.profile.as_video_stream_profile()
        intrinsics = video_profile.get_intrinsics()
        image = np.asanyarray(frame.get_data()).copy()
        hfov, vfov = _field_of_view(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
        )
        return Capture(
            image=image,
            width=int(intrinsics.width),
            height=int(intrinsics.height),
            hfov_deg=hfov,
            vfov_deg=vfov,
            fx=float(intrinsics.fx),
            ppx=float(intrinsics.ppx),
        )
    finally:
        try:
            pipeline.stop()
        except RuntimeError:
            pass


def _put_label(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def _make_comparison(standard: Capture, wide: Capture) -> np.ndarray:
    cell_width = max(standard.width, wide.width)
    image_height = max(standard.height, wide.height)
    header_height = 76
    footer_height = 46
    gap = 16
    canvas = np.full(
        (header_height + image_height + footer_height, cell_width * 2 + gap, 3),
        24,
        dtype=np.uint8,
    )

    standard_x = (cell_width - standard.width) // 2
    standard_y = header_height + (image_height - standard.height) // 2
    wide_cell_x = cell_width + gap
    wide_x = wide_cell_x + (cell_width - wide.width) // 2
    wide_y = header_height + (image_height - wide.height) // 2
    canvas[
        standard_y : standard_y + standard.height,
        standard_x : standard_x + standard.width,
    ] = standard.image
    canvas[wide_y : wide_y + wide.height, wide_x : wide_x + wide.width] = wide.image

    _put_label(canvas, "CURRENT RGB 4:3", (16, 28), (255, 255, 255))
    _put_label(
        canvas,
        f"{standard.width}x{standard.height}  FOV {standard.hfov_deg:.1f} x {standard.vfov_deg:.1f} deg",
        (16, 58),
        (120, 220, 255),
    )
    _put_label(canvas, "WIDE RGB 16:9", (wide_cell_x + 16, 28), (255, 255, 255))
    _put_label(
        canvas,
        f"{wide.width}x{wide.height}  FOV {wide.hfov_deg:.1f} x {wide.vfov_deg:.1f} deg",
        (wide_cell_x + 16, 58),
        (120, 255, 120),
    )

    # Project the horizontal limits of the 4:3 profile into the wide profile.
    narrow_half_angle = math.radians(standard.hfov_deg / 2.0)
    half_width = wide.fx * math.tan(narrow_half_angle)
    left = int(round(wide_x + wide.ppx - half_width))
    right = int(round(wide_x + wide.ppx + half_width))
    left = max(wide_x, min(wide_x + wide.width - 1, left))
    right = max(wide_x, min(wide_x + wide.width - 1, right))
    cv2.rectangle(
        canvas,
        (left, wide_y),
        (right, wide_y + wide.height - 1),
        (80, 255, 80),
        2,
    )
    _put_label(
        canvas,
        "green box = horizontal area retained by current 4:3 mode",
        (wide_cell_x + 16, header_height + image_height + 30),
        (120, 255, 120),
    )
    _put_label(
        canvas,
        "Press Q or Esc to close",
        (16, header_height + image_height + 30),
        (210, 210, 210),
    )
    return canvas


def _select_device(rs: Any, requested_serial: str | None) -> tuple[str, str]:
    devices = list(rs.context().query_devices())
    candidates: list[tuple[str, str]] = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        if "RealSense" in name:
            candidates.append((serial, name))

    if requested_serial:
        for serial, name in candidates:
            if serial == requested_serial:
                return serial, name
        available = ", ".join(serial for serial, _ in candidates) or "none"
        raise RuntimeError(f"D435 serial {requested_serial} was not found; available RealSense serials: {available}")

    if not candidates:
        raise RuntimeError("no RealSense camera was found")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the D435 RGB 640x480 crop with the wider 640x360 mode.",
    )
    parser.add_argument("--serial", help="D435 serial; defaults to the first connected RealSense camera")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--exposure", type=float, default=141.0)
    parser.add_argument("--white-balance", type=float, default=3780.0)
    parser.add_argument("--output", type=Path, help="comparison JPEG path; defaults to /tmp")
    parser.add_argument("--no-window", action="store_true", help="save without opening an OpenCV window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("ERROR: pyrealsense2 is not installed in this Python environment", file=sys.stderr)
        return 2

    try:
        serial, name = _select_device(rs, args.serial)
        print(f"D435: {name} | serial={serial}")
        print("Capturing current 640x480 mode; keep the camera still ...")
        standard = _capture_mode(
            rs,
            serial,
            640,
            480,
            args.fps,
            args.warmup_frames,
            args.exposure,
            args.white_balance,
        )
        time.sleep(0.35)
        print("Capturing wide 640x360 mode ...")
        wide = _capture_mode(
            rs,
            serial,
            640,
            360,
            args.fps,
            args.warmup_frames,
            args.exposure,
            args.white_balance,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Close the Fabric-DROID collection UI and any other camera application, then retry.", file=sys.stderr)
        return 1

    comparison = _make_comparison(standard, wide)
    output = args.output or Path(f"/tmp/d435_fov_compare_{serial}.jpg")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), comparison):
        print(f"ERROR: failed to save {output}", file=sys.stderr)
        return 1

    print(
        f"640x480 FOV: {standard.hfov_deg:.2f} x {standard.vfov_deg:.2f} deg\n"
        f"640x360 FOV: {wide.hfov_deg:.2f} x {wide.vfov_deg:.2f} deg\n"
        f"Saved: {output}"
    )

    if not args.no_window and os.environ.get("DISPLAY"):
        cv2.namedWindow("D435 RGB FOV comparison", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("D435 RGB FOV comparison", min(comparison.shape[1], 1500), min(comparison.shape[0], 780))
        cv2.imshow("D435 RGB FOV comparison", comparison)
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
    elif not args.no_window:
        print("DISPLAY is not set; the comparison was saved without opening a window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
