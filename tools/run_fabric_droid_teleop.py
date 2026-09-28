#!/usr/bin/env python3
"""Launch DROID teleoperation with the Fabric-DROID sidecar.

This is intentionally guarded and is never invoked by automated tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.capture.droid_hook import make_droid_sidecar_hook_factory
from fabric_droid.capture.recorder import RecorderOptions
from fabric_droid.io_utils import git_commit
from fabric_droid.schemas import (
    CANONICAL_TASK_INSTRUCTION,
    EpisodeMetadata,
    ForceCalibration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    parser.add_argument(
        "--destination-tray",
        choices=("target_tray",),
        default="target_tray",
    )
    parser.add_argument("--swatch-uid", required=True)
    parser.add_argument("--split", choices=("train", "validation", "heldout_test"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--source-slot", default="source_slot_1")
    parser.add_argument("--start-pose-bucket", default="ready")
    parser.add_argument("--d435-serial", required=True)
    parser.add_argument("--wrist-source", required=True)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--gelsight-source", required=True)
    parser.add_argument("--gelsight-serial", required=True)
    parser.add_argument("--ati-endpoint", default="tcp://192.168.1.20:5555")
    parser.add_argument("--ati-serial", required=True)
    parser.add_argument("--calibration-id", default="uncalibrated")
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--left-controller", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.enable_robot or not args.preflight_confirmed:
        print(
            "Refusing to construct RobotEnv. Complete the pilot checklist, then pass both "
            "--enable-robot and --preflight-confirmed.",
            file=sys.stderr,
        )
        return 2

    # Robot imports happen only after both explicit safety acknowledgements.
    from droid.controllers.oculus_controller import VRPolicy
    from droid.misc.parameters import robot_serial_number, robot_type
    from droid.robot_env import RobotEnv
    from droid.user_interface.data_collector import DataCollecter
    from droid.user_interface.gui import RobotGUI

    prompt = CANONICAL_TASK_INSTRUCTION

    def metadata_builder(episode_dir: Path, droid_info: dict[str, object]) -> EpisodeMetadata:
        return EpisodeMetadata(
            episode_id=episode_dir.name,
            task_instruction=prompt,
            destination_tray=args.destination_tray,
            source_slot=args.source_slot,
            swatch_uid=args.swatch_uid,
            split=args.split,
            start_pose_bucket=args.start_pose_bucket,
            success=True,
            failure_reason="",
            operator=args.operator,
            session_id=args.session_id,
            robot_id=f"{robot_type}-{robot_serial_number}",
            camera_serials={
                "exterior_image_1_left": args.d435_serial,
                "wrist_image_left": args.wrist_serial,
            },
            gelsight_serial=args.gelsight_serial,
            ati_serial=args.ati_serial,
            calibration_id=calibration.calibration_id,
            software_git_commits={
                "droid": git_commit(Path(__file__).resolve().parents[1]),
            },
            robot_motion_enabled=True,
        )

    options = RecorderOptions(
        output_dir=Path("/tmp/unused-fabric-droid-sidecar-root"),
        dry_run=False,
        robot_disabled=True,
        record_only=True,
        d435_serial=args.d435_serial,
        wrist_source=args.wrist_source,
        wrist_serial=args.wrist_serial,
        gelsight_source=args.gelsight_source,
        gelsight_serial=args.gelsight_serial,
        ati_endpoint=args.ati_endpoint,
    )

    def hold_gripper(action: object) -> np.ndarray:
        safe = np.asarray(action).copy()
        safe[-1] = 0.0
        return safe

    def abort(reason: str) -> None:
        print(f"ATI SAFETY ABORT: {reason}", file=sys.stderr, flush=True)

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

    hook_factory = make_droid_sidecar_hook_factory(
        options,
        metadata_builder,
        block_closing=hold_gripper,
        abort_episode=abort,
        calibration=calibration,
    )
    env = RobotEnv()
    controller = VRPolicy(right_controller=not args.left_controller)
    collector = DataCollecter(env=env, controller=controller, lifecycle_hook_factory=hook_factory)
    print(
        f"Fabric-DROID armed at {time.asctime()}. F9=probe_complete, F10=release_time. "
        "DROID is the sole robot-control owner.",
        flush=True,
    )
    RobotGUI(robot=collector, right_controller=not args.left_controller)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
