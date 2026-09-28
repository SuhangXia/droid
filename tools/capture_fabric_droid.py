#!/usr/bin/env python3
"""Record one Fabric-DROID episode without enabling robot motion."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.capture.recorder import FabricEpisodeRecorder, RecorderOptions, default_metadata
from fabric_droid.io_utils import git_commit
from fabric_droid.schemas import ForceCalibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument(
        "--destination-tray",
        choices=("target_tray",),
        default="target_tray",
    )
    parser.add_argument("--swatch-uid", required=True)
    parser.add_argument("--split", choices=("train", "validation", "heldout_test"), default="train")
    parser.add_argument("--session-id")
    parser.add_argument("--episode-id")
    parser.add_argument("--operator", default="unknown")
    parser.add_argument("--source-slot", default="source_slot_1")
    parser.add_argument("--start-pose-bucket", default="ready")
    parser.add_argument("--robot-id", default="robot-disabled")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic synthetic sensors")
    parser.add_argument("--record-only", action="store_true", help="Capture real sensors but never connect to the robot")
    parser.add_argument("--robot-disabled", action="store_true", help="Assert that no robot motion is allowed")
    parser.add_argument("--d435-serial", default="")
    parser.add_argument("--wrist-kind", choices=("uvc", "d435"), default="uvc")
    parser.add_argument("--wrist-source", default="0")
    parser.add_argument("--wrist-serial", default="wrist-uvc")
    parser.add_argument("--gelsight-source", default="1")
    parser.add_argument("--gelsight-serial", default="gelsight-left")
    parser.add_argument("--ati-endpoint", default="tcp://192.168.1.20:5555")
    parser.add_argument("--no-ati", action="store_true", help="Record an explicit unavailable ATI stream")
    parser.add_argument("--ati-serial", default="ati-nano17")
    parser.add_argument("--calibration-id", default="uncalibrated")
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument(
        "--interactive-events",
        action="store_true",
        help="Read event names from stdin; shortcuts: p=probe_complete, r=release_time",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = default_metadata(args.destination_tray, args.swatch_uid, args.split, args.session_id, args.episode_id)
    metadata = replace(
        metadata,
        source_slot=args.source_slot,
        start_pose_bucket=args.start_pose_bucket,
        operator=args.operator,
        robot_id=args.robot_id,
        camera_serials={
            "exterior_image_1_left": args.d435_serial or "d435",
            "wrist_image_left": args.wrist_serial,
        },
        gelsight_serial=args.gelsight_serial,
        ati_serial=args.ati_serial,
        calibration_id=args.calibration_id,
        software_git_commits={"droid": git_commit(Path(__file__).resolve().parents[1])},
    )
    options = RecorderOptions(
        output_dir=args.output_dir,
        duration_sec=args.duration_sec,
        dry_run=args.dry_run,
        record_only=args.record_only or args.dry_run,
        robot_disabled=args.robot_disabled or args.record_only or args.dry_run,
        d435_serial=args.d435_serial,
        wrist_kind=args.wrist_kind,
        wrist_source=args.wrist_source,
        wrist_serial=args.wrist_serial,
        gelsight_source=args.gelsight_source,
        gelsight_serial=args.gelsight_serial,
        ati_enabled=not args.no_ati,
        ati_endpoint=args.ati_endpoint,
    )
    calibration = ForceCalibration(calibration_id=args.calibration_id)
    if args.calibration_json is not None:
        payload = json.loads(args.calibration_json.read_text(encoding="utf-8"))
        calibration = ForceCalibration(
            calibration_id=payload["calibration_id"],
            T_ati_to_gripper=payload.get("T_ati_to_gripper"),
            gripper_normal_axis=payload.get("gripper_normal_axis"),
            force_sign=payload.get("force_sign"),
            bias_wrench=payload.get("bias_wrench", [0.0] * 6),
        )
        metadata = replace(metadata, calibration_id=calibration.calibration_id)
    episode = FabricEpisodeRecorder(options, metadata, calibration).record_for_duration(
        interactive_events=args.interactive_events
    )
    print(json.dumps({"episode": str(episode), "robot_motion_enabled": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
