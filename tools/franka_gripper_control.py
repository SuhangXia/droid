#!/usr/bin/env python3
"""Read or command the Polymetis gripper using millimetre CLI units."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fabric_droid.robot.gripper_control import (
    MAX_SAFE_CLOSEDNESS,
    GripperControlError,
    GripperMotion,
    closedness_for_width,
    move_gripper,
    read_gripper,
    width_for_closedness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guarded Franka/Polymetis gripper width control. "
            "Without --enable-gripper it only reads state."
        )
    )
    parser.add_argument("--robot-ip", default="192.168.0.116")
    parser.add_argument("--gripper-port", type=int, default=50052)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--width-mm", type=float, help="target total finger opening in millimetres")
    target.add_argument(
        "--closedness-percent",
        type=float,
        help="target closure percentage; hard-limited to 98%%",
    )
    target.add_argument("--open", action="store_true", help="open to metadata max_width")
    parser.add_argument("--speed-mm-s", type=float, default=10.0)
    parser.add_argument("--force-n", type=float, default=5.0)
    parser.add_argument(
        "--result-json",
        type=Path,
        help="optional machine-readable result path for an authorized UI workflow",
    )
    parser.add_argument("--enable-gripper", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    parser.add_argument(
        "--non-interactive-ui",
        action="store_true",
        help="skip typed confirmation for an already-authorized collection UI action",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        help="send the bounded command without waiting for width convergence",
    )
    return parser.parse_args()


def _confirmation(target_mm: float, current_mm: float) -> bool:
    direction = "CLOSE" if target_mm < current_mm else "OPEN"
    print(
        f"\nREAL GRIPPER COMMAND: {direction} from {current_mm:.2f}mm to {target_mm:.2f}mm.\n"
        "Verify both fingers and the full pinch region are clear. "
        "Type ENABLE GRIPPER exactly to continue:",
        flush=True,
    )
    try:
        return input("> ").strip() == "ENABLE GRIPPER"
    except EOFError:
        return False


def main() -> int:
    args = parse_args()
    if not 1 <= args.gripper_port <= 65535:
        print("INVALID CONFIG: gripper port must be within [1, 65535]", file=sys.stderr)
        return 2

    from polymetis import GripperInterface

    gripper = GripperInterface(ip_address=args.robot_ip, port=args.gripper_port)
    try:
        state = read_gripper(gripper)
    except (GripperControlError, RuntimeError) as exc:
        print(f"GRIPPER FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"robot_ip": args.robot_ip, "gripper": state}, indent=2), flush=True)

    wants_motion = args.width_mm is not None or args.closedness_percent is not None or args.open
    if not wants_motion:
        print("READ ONLY: no gripper command was sent.", flush=True)
        return 0
    if not args.enable_gripper or not args.preflight_confirmed:
        print(
            "Refusing gripper motion. Pass both --enable-gripper and "
            "--preflight-confirmed after clearing the pinch region.",
            file=sys.stderr,
        )
        return 2

    if args.open:
        target_mm = float(state["max_width_mm"])
    elif args.closedness_percent is not None:
        try:
            target_mm = 1000.0 * width_for_closedness(
                float(state["max_width_m"]),
                float(args.closedness_percent) / 100.0,
            )
        except GripperControlError as exc:
            print(f"INVALID GRIPPER COMMAND: {exc}", file=sys.stderr)
            return 2
    else:
        target_mm = float(args.width_mm)
    motion = GripperMotion(
        width_m=target_mm / 1000.0,
        speed_m_s=args.speed_mm_s / 1000.0,
        force_n=args.force_n,
    )
    try:
        motion.validate(max_width_m=float(state["max_width_m"]))
    except GripperControlError as exc:
        print(f"INVALID GRIPPER COMMAND: {exc}", file=sys.stderr)
        return 2
    if not args.non_interactive_ui and not _confirmation(
        target_mm,
        float(state["width_mm"]),
    ):
        print("Confirmation did not match; no gripper command was sent.", file=sys.stderr)
        return 2

    try:
        result = move_gripper(
            gripper,
            motion,
            verify_width=not args.non_blocking,
            blocking=not args.non_blocking,
            timeout_s=12.0,
            tolerance_m=0.001,
        )
    except (GripperControlError, RuntimeError) as exc:
        print(f"GRIPPER FAILED: {exc}", file=sys.stderr)
        return 1
    if args.open and not args.non_blocking:
        after_width_m = float(result["after"]["width_m"])
        max_width_m = float(result["after"]["max_width_m"])
        if after_width_m < max_width_m - 0.003:
            print(
                "GRIPPER FAILED: blocking open command ended but measured width "
                f"{after_width_m * 1000.0:.2f}mm is more than 3mm below "
                f"maximum {max_width_m * 1000.0:.2f}mm",
                file=sys.stderr,
            )
            return 1
    payload = {
        "safety": {
            "max_closedness_percent": MAX_SAFE_CLOSEDNESS * 100.0,
            "minimum_opening_mm": 1000.0
            * width_for_closedness(
                float(state["max_width_m"]), MAX_SAFE_CLOSEDNESS
            ),
        },
        "command": {
            **vars(motion),
            "closedness_percent": 100.0
            * closedness_for_width(float(state["max_width_m"]), motion.width_m),
        },
        "completion": "command_sent" if args.non_blocking else "width_verified",
        "result": result,
    }
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
