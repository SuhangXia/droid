#!/usr/bin/env python3
"""Run guarded Franka hand guiding and atomically remember a home pose."""

from __future__ import annotations

import argparse
import json
import select
import sys
import threading
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.robot.freedrive_home import (
    FreedriveError,
    LowImpedanceConfig,
    preflight_robot,
    run_low_impedance_freedrive,
    save_home_snapshot,
    wait_for_stationary_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOME_PATH = REPO_ROOT / "configs" / "robot" / "franka_home_pose.json"


def _six(values: Sequence[str]) -> tuple[float, float, float, float, float, float]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 6:
        raise argparse.ArgumentTypeError("expected exactly six values")
    return parsed  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guarded low-Cartesian-impedance Franka hand guiding. "
            "Press Enter to atomically save the current flange/joint configuration as home."
        )
    )
    parser.add_argument("--robot-ip", default="192.168.0.116")
    parser.add_argument("--home-pose-file", type=Path, default=DEFAULT_HOME_PATH)
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="read preflight state and exit without starting a policy")
    parser.add_argument(
        "--non-interactive-ui",
        action="store_true",
        help="skip terminal prompts and accept SAVE_HOME lines on stdin (for the collection UI)",
    )
    parser.add_argument("--overwrite-home", action="store_true")
    parser.add_argument("--follow-hz", type=float, default=15.0)
    parser.add_argument("--max-duration-sec", type=float, default=300.0)
    parser.add_argument(
        "--stiffness",
        nargs=6,
        metavar=("KX", "KY", "KZ", "KRX", "KRY", "KRZ"),
        default=("20", "20", "20", "2", "2", "2"),
    )
    parser.add_argument(
        "--damping",
        nargs=6,
        metavar=("DX", "DY", "DZ", "DRX", "DRY", "DRZ"),
        default=("7", "7", "7", "0.8", "0.8", "0.8"),
    )
    return parser.parse_args()


def _confirmation() -> bool:
    print(
        "\nThis will start LOW Cartesian impedance on the real Franka.\n"
        "Verify: robot unlocked, FCI active, emergency stop reachable, workspace clear,\n"
        "and support the arm before enabling. Type ENABLE FREEDRIVE to continue:",
        flush=True,
    )
    try:
        return input("> ").strip() == "ENABLE FREEDRIVE"
    except EOFError:
        return False


def main() -> int:
    args = parse_args()
    config = LowImpedanceConfig(
        stiffness=_six(args.stiffness),
        damping=_six(args.damping),
        follow_hz=args.follow_hz,
        max_duration_sec=args.max_duration_sec,
    )
    config.validate()

    if args.home_pose_file.exists() and not args.overwrite_home and not args.check_only:
        print(
            f"Refusing to overwrite existing home pose: {args.home_pose_file}\n"
            "Use --overwrite-home only after verifying the existing file.",
            file=sys.stderr,
        )
        return 2

    if not args.check_only and (not args.enable_robot or not args.preflight_confirmed):
        print(
            "Refusing to start robot control. Pass both --enable-robot and "
            "--preflight-confirmed after completing the physical safety check.",
            file=sys.stderr,
        )
        return 2

    # Robot dependencies are imported only after CLI validation.
    import torch
    from polymetis import RobotInterface

    robot = RobotInterface(ip_address=args.robot_ip)
    if args.non_interactive_ui:
        preflight = wait_for_stationary_preflight(robot, timeout_sec=3.0)
    else:
        preflight = preflight_robot(robot)
    print(json.dumps({"robot_ip": args.robot_ip, "preflight": preflight}, indent=2), flush=True)
    if args.check_only:
        print("CHECK ONLY: no control policy was started.", flush=True)
        return 0
    if not args.non_interactive_ui and not _confirmation():
        print("Confirmation did not match; no control policy was started.", file=sys.stderr)
        return 2

    if args.non_interactive_ui:
        print(
            "UI CONTROL MODE: low impedance will start; send SAVE_HOME on stdin to save and stop.",
            flush=True,
        )

        def should_save() -> bool:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return False
            return sys.stdin.readline().strip() == "SAVE_HOME"

    else:
        save_requested = threading.Event()

        def wait_for_enter() -> None:
            try:
                input(
                    "\nLOW IMPEDANCE ACTIVE. Support and hand-guide the arm.\n"
                    "Press Enter to SAVE the current pose as home and exit.\n"
                )
                save_requested.set()
            except EOFError:
                return

        input_thread = threading.Thread(target=wait_for_enter, daemon=True)
        input_thread.start()
        should_save = save_requested.is_set

    last_print_second = -1

    def status(tick: dict[str, object]) -> None:
        nonlocal last_print_second
        second = int(float(tick["elapsed_sec"]))
        if second != last_print_second:
            last_print_second = second
            print(
                f"[freedrive] elapsed={second}s position_m={tick['position_m']}",
                flush=True,
            )

    try:
        snapshot = run_low_impedance_freedrive(
            robot,
            config=config,
            should_save=should_save,
            tensor_factory=lambda values: torch.tensor(values, dtype=torch.float32),
            on_tick=status,
            preflight_timeout_sec=3.0 if args.non_interactive_ui else 0.0,
        )
    except KeyboardInterrupt:
        print("\nCancelled with Ctrl+C; home was not saved.", file=sys.stderr)
        return 130
    except FreedriveError as exc:
        print(f"FREEDRIVE FAILED: {exc}", file=sys.stderr)
        return 1

    if snapshot is None:
        print("Freedrive timed out; home was not saved.", file=sys.stderr)
        return 1
    payload = save_home_snapshot(
        args.home_pose_file,
        snapshot,
        robot_ip=args.robot_ip,
        source="tools/franka_freedrive_home.py",
        config=config,
        overwrite=args.overwrite_home,
    )
    print(f"Saved home pose atomically: {args.home_pose_file}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
