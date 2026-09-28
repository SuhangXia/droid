#!/usr/bin/env python3
"""Return a Franka slowly to a Fabric-DROID saved joint home."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fabric_droid.robot.go_home import (
    GoHomeError,
    home_preflight,
    load_saved_home,
    run_go_home,
)


DEFAULT_HOME_PATH = REPO_ROOT / "configs" / "robot" / "franka_home_pose.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guarded minimum-jerk joint-space return to the saved Fabric-DROID home."
    )
    parser.add_argument("--robot-ip", default="192.168.0.116")
    parser.add_argument("--home-pose-file", type=Path, default=DEFAULT_HOME_PATH)
    parser.add_argument("--time-to-go-sec", type=float, default=10.0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    return parser.parse_args()


def _confirmation(preflight: dict[str, object], time_to_go_sec: float) -> bool:
    print(
        "\nREAL FRANKA JOINT-SPACE HOME MOTION WILL START.\n"
        f"Maximum joint displacement: {float(preflight['max_abs_joint_delta_rad']):.3f} rad\n"
        f"Planned duration: {time_to_go_sec:.1f} s\n"
        "This path has no environment collision checking. Clear the complete swept volume,\n"
        "keep the physical emergency stop reachable, and type MOVE TO HOME exactly:",
        flush=True,
    )
    try:
        return input("> ").strip() == "MOVE TO HOME"
    except EOFError:
        return False


def main() -> int:
    args = parse_args()
    try:
        home = load_saved_home(args.home_pose_file, expected_robot_ip=args.robot_ip)
    except GoHomeError as exc:
        print(f"HOME REFUSED: {exc}", file=sys.stderr)
        return 2
    if not args.check_only and (not args.enable_robot or not args.preflight_confirmed):
        print(
            "Refusing home motion. Pass both --enable-robot and --preflight-confirmed "
            "after clearing the full swept volume.",
            file=sys.stderr,
        )
        return 2

    import torch
    from polymetis import RobotInterface

    robot = RobotInterface(ip_address=args.robot_ip)
    try:
        preflight = home_preflight(robot, home)
    except Exception as exc:
        print(f"HOME PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"robot_ip": args.robot_ip, "home_preflight": preflight}, indent=2), flush=True)
    if args.check_only:
        print("CHECK ONLY: no control policy was started.", flush=True)
        return 0
    if not _confirmation(preflight, args.time_to_go_sec):
        print("Confirmation did not match; no control policy was started.", file=sys.stderr)
        return 2

    try:
        result = run_go_home(
            robot,
            home,
            time_to_go_sec=args.time_to_go_sec,
            tensor_factory=lambda values: torch.tensor(values, dtype=torch.float32),
        )
    except KeyboardInterrupt:
        print("\nCtrl+C: owned home policy terminated.", file=sys.stderr)
        return 130
    except GoHomeError as exc:
        print(f"HOME FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

