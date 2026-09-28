#!/usr/bin/env python3
"""Open both D435s and GelSight concurrently and report actual frame health."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.sensors.camera import CameraSpec, CameraStream
from fabric_droid.ui.devices import discover_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/fabric_droid_camera_probe"))
    parser.add_argument("--skip-gelsight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = discover_devices()
    if len(inventory.realsense) < 2:
        raise RuntimeError(f"expected two D435 devices, found {len(inventory.realsense)}")
    gelsight = next(
        (
            device
            for device in inventory.uvc
            if "gelsight" in device.name.lower()
            or "arducam" in device.name.lower()
            or device.serial == "2DWF0RJM"
        ),
        None,
    )
    if gelsight is None and not args.skip_gelsight:
        raise RuntimeError(
            "GelSight was not found among UVC devices: "
            + ", ".join(f"{device.name} ({device.serial})" for device in inventory.uvc)
        )
    first, second = inventory.realsense[:2]
    print(
        json.dumps(
            {
                "inventory": inventory.to_dict(),
                "opening": {
                    "d435_a": first.to_dict(),
                    "d435_b": second.to_dict(),
                    "gelsight": None if gelsight is None or args.skip_gelsight else gelsight.to_dict(),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    specs = [
        CameraSpec(
            "d435_a",
            "d435",
            first.serial,
            first.serial,
            640,
            480,
            30,
            require_usb3=True,
        ),
        CameraSpec(
            "d435_b",
            "d435",
            second.serial,
            second.serial,
            640,
            480,
            30,
            require_usb3=True,
        ),
    ]
    if not args.skip_gelsight:
        assert gelsight is not None
        specs.append(CameraSpec("gelsight", "uvc", gelsight.source, gelsight.serial, 3280, 2464, 25, "MJPG"))
    streams = [
        CameraStream(
            spec,
            max_buffer_frames=2,
            frame_output_size=(640, 480) if spec.name == "gelsight" else None,
        )
        for spec in specs
    ]
    try:
        for stream in streams:
            stream.start()
        warmup_deadline = time.monotonic() + 5.0
        while time.monotonic() < warmup_deadline:
            failed = [stream for stream in streams if stream.error is not None]
            if failed:
                raise RuntimeError(f"{failed[0].name} failed: {failed[0].error}")
            if all(stream.snapshot()["count"] >= 5 for stream in streams):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("camera warmup timed out before five frames per stream")
        monitor_start_ns = time.monotonic_ns()
        for stream in streams:
            stream.reset_gap_monitor(monitor_start_ns)
        deadline = time.monotonic() + args.duration_sec
        while time.monotonic() < deadline:
            failed = [stream for stream in streams if stream.error is not None]
            if failed:
                raise RuntimeError(f"{failed[0].name} failed: {failed[0].error}")
            time.sleep(0.05)
    finally:
        for stream in reversed(streams):
            try:
                stream.stop(timeout=3.0)
            except Exception:
                pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for probe snapshots") from exc
    snapshots: dict[str, object] = {}
    for stream in streams:
        snapshots[stream.name] = stream.snapshot()
        frame = stream.latest_frame()
        if frame is not None:
            cv2.imwrite(str(args.output_dir / f"{stream.name}.jpg"), frame.image_bgr)
    minimum_hz = {
        "d435_a": 24.0,
        "d435_b": 24.0,
        # The Octopi reference sessions from this exact GelSight firmware
        # consistently measure about 18.75 Hz when 25 FPS is requested.
        "gelsight": 17.5,
    }
    rate_pass = {
        name: (
            snapshot["measured_hz"] is not None
            and float(snapshot["measured_hz"]) >= minimum_hz[name]
        )
        for name, snapshot in snapshots.items()
    }
    report = {
        "pass": (
            all(snapshot["healthy"] for snapshot in snapshots.values())
            and all(rate_pass.values())
        ),
        "inventory": inventory.to_dict(),
        "assignment_for_visual_confirmation": {
            "candidate_wrist": first.serial,
            "candidate_third_person": second.serial,
            "gelsight": None if gelsight is None or args.skip_gelsight else gelsight.serial,
        },
        "streams": snapshots,
        "minimum_hz": {
            name: minimum_hz[name] for name in snapshots
        },
        "rate_pass": rate_pass,
        "snapshots": str(args.output_dir),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
