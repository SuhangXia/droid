#!/usr/bin/env python3
"""Preview or run guarded right-hand Quest teleoperation on a Franka."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "droid" / "oculus_reader"))

from fabric_droid.robot.freedrive_home import (
    FreedriveError,
    preflight_robot,
    wait_for_stationary_preflight,
)
from fabric_droid.capture.robot_trajectory import RobotTelemetryBuffer
from fabric_droid.robot.gripper_control import GripperControlError, read_gripper
from fabric_droid.robot.quest_reader import TimestampedRightQuestSource
from fabric_droid.robot.quest_teleop import (
    QuestGripperConfig,
    QuestGripperController,
    QuestTeleopConfig,
    QuestTeleopError,
    right_trigger_closedness,
    run_right_quest_teleop,
    validate_quest_frame,
)


def _three(values: list[str]) -> tuple[float, float, float]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("expected exactly three values")
    return parsed  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safe right-hand Quest teleoperation. Default behavior refuses robot motion; "
            "use --preview-only first, then --check-only."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preview-only", action="store_true", help="Quest stream only; never connect to Franka")
    mode.add_argument("--check-only", action="store_true", help="Quest plus read-only Franka preflight; no policy")
    parser.add_argument("--robot-ip", default="192.168.0.116")
    parser.add_argument("--quest-ip", default=None, help="omit for the currently connected USB Quest")
    parser.add_argument("--quest-wait-sec", type=float, default=10.0)
    parser.add_argument("--enable-robot", action="store_true")
    parser.add_argument("--preflight-confirmed", action="store_true")
    parser.add_argument(
        "--non-interactive-ui",
        action="store_true",
        help="skip the typed confirmation for an already-authorized collection UI launch",
    )
    parser.add_argument(
        "--telemetry-output",
        type=Path,
        default=None,
        help="atomically write real 30 Hz robot telemetry NPZ for trajectory.h5 finalization",
    )
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=120.0,
        help="session limit in seconds; exactly 0 means no total-duration timeout",
    )
    parser.add_argument("--max-linear-speed", type=float, default=0.05)
    parser.add_argument("--max-angular-speed-deg", type=float, default=20.0)
    parser.add_argument("--controller-timeout-sec", type=float, default=0.25)
    parser.add_argument(
        "--enable-gripper",
        action="store_true",
        help="enable DROID-compatible right-index-trigger gripper control while RG is held",
    )
    parser.add_argument("--gripper-port", type=int, default=50052)
    parser.add_argument("--gripper-speed-mm-s", type=float, default=20.0)
    parser.add_argument("--gripper-force-n", type=float, default=10.0)
    parser.add_argument("--gripper-command-hz", type=float, default=15.0)
    parser.add_argument(
        "--gripper-max-closedness",
        type=float,
        default=0.98,
        help="maximum normalized closure from rightTrig; default 0.98 keeps about 1.62mm open",
    )
    parser.add_argument(
        "--workspace-min",
        nargs=3,
        default=("0.20", "-0.60", "0.05"),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--workspace-max",
        nargs=3,
        default=("0.80", "0.60", "0.90"),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--workspace-recovery-margin",
        type=float,
        default=0.02,
        help=(
            "maximum initial distance outside the workspace that may be recovered; "
            "outside this margin startup is refused (default: 0.02 m)"
        ),
    )
    parser.add_argument(
        "--disable-workspace-limit",
        action="store_true",
        help=(
            "disable all software Cartesian workspace checks/clamping/recovery; "
            "deadman, rate, tracking and robot watchdogs remain enabled"
        ),
    )
    return parser.parse_args()


def _config(args: argparse.Namespace) -> QuestTeleopConfig:
    config = QuestTeleopConfig(
        control_hz=args.rate,
        max_duration_sec=args.max_duration_sec,
        controller_timeout_sec=args.controller_timeout_sec,
        max_translation_speed_m_s=args.max_linear_speed,
        max_rotation_speed_rad_s=math.radians(args.max_angular_speed_deg),
        workspace_min_m=_three(args.workspace_min),
        workspace_max_m=_three(args.workspace_max),
        workspace_recovery_margin_m=args.workspace_recovery_margin,
        enforce_workspace_limits=not args.disable_workspace_limit,
    )
    config.validate()
    return config


def _confirmation(*, gripper_enabled: bool) -> bool:
    gripper_check = (
        "  4) gripper pinch region is clear; RIGHT index trigger and RG are RELEASED.\n"
        if gripper_enabled
        else "  4) RIGHT side-grip RG is RELEASED; index trigger/gripper is disabled.\n"
    )
    print(
        "\nREAL FRANKA TELEOP WILL START.\n"
        "Verify all of the following now:\n"
        "  1) Franka desk is unlocked/FCI active and no other controller is running;\n"
        "  2) workspace is clear and another person is not near the arm;\n"
        "  3) physical emergency stop is in your free hand/reachable;\n"
        f"{gripper_check}"
        "Type ENABLE QUEST TELEOP exactly to continue:",
        flush=True,
    )
    try:
        return input("> ").strip() == "ENABLE QUEST TELEOP"
    except EOFError:
        return False


def _frame_summary(frame: object, *, gripper_enabled: bool = False) -> dict[str, object]:
    pose = frame.pose  # type: ignore[attr-defined]
    buttons = frame.buttons  # type: ignore[attr-defined]
    summary = {
        "sequence": frame.sequence,  # type: ignore[attr-defined]
        "right_position": [round(float(value), 4) for value in pose[:3, 3]],
        "RG_deadman": bool(buttons.get("RG", False)),
        "RJ_rezero": bool(buttons.get("RJ", False)),
        "B_exit": bool(buttons.get("B", False)),
        "gripper_enabled": gripper_enabled,
    }
    if "rightTrig" in buttons:
        summary["right_index_trigger"] = round(right_trigger_closedness(buttons), 4)
    return summary


def _preview(
    source: TimestampedRightQuestSource,
    config: QuestTeleopConfig,
    *,
    gripper_enabled: bool,
) -> int:
    print(
        "QUEST PREVIEW ONLY: Franka is not connected. Test RG/RJ/B now; "
        "this mode cannot move the robot.",
        flush=True,
    )
    deadline = (
        None
        if config.max_duration_sec == 0.0
        else time.monotonic() + config.max_duration_sec
    )
    last_sequence = -1
    last_print = 0.0
    while deadline is None or time.monotonic() < deadline:
        frame = source.latest_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        validate_quest_frame(
            frame,
            now_ns=time.monotonic_ns(),
            timeout_sec=config.controller_timeout_sec,
        )
        if frame.buttons.get("B", False):
            print("B pressed: preview exited.", flush=True)
            return 0
        now = time.monotonic()
        if frame.sequence != last_sequence and now - last_print >= 0.2:
            print(
                json.dumps(
                    _frame_summary(frame, gripper_enabled=gripper_enabled),
                    ensure_ascii=False,
                ),
                flush=True,
            )
            last_sequence = frame.sequence
            last_print = now
        time.sleep(0.01)
    print("Preview timeout reached; no robot connection was made.", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    telemetry_buffer: Optional[RobotTelemetryBuffer] = None
    try:
        config = _config(args)
    except (ValueError, QuestTeleopError) as exc:
        print(f"INVALID CONFIG: {exc}", file=sys.stderr)
        return 2
    if args.telemetry_output is not None:
        if args.preview_only or args.check_only:
            print("INVALID CONFIG: telemetry output requires real teleoperation", file=sys.stderr)
            return 2
        if not args.enable_gripper:
            print(
                "INVALID CONFIG: telemetry output requires --enable-gripper so DROID "
                "gripper state/action are real",
                file=sys.stderr,
            )
            return 2
        if args.telemetry_output.exists():
            print(
                f"INVALID CONFIG: refusing to overwrite telemetry {args.telemetry_output}",
                file=sys.stderr,
            )
            return 2

    if not args.preview_only and not args.check_only and (
        not args.enable_robot or not args.preflight_confirmed
    ):
        print(
            "Refusing robot motion. First run --preview-only and --check-only. "
            "Real control requires both --enable-robot and --preflight-confirmed.",
            file=sys.stderr,
        )
        return 2

    source: Optional[TimestampedRightQuestSource] = None
    try:
        source = TimestampedRightQuestSource(ip_address=args.quest_ip)
        first_frame = source.wait_for_frame(args.quest_wait_sec)
        validate_quest_frame(
            first_frame,
            now_ns=time.monotonic_ns(),
            timeout_sec=config.controller_timeout_sec,
        )
        print(
            json.dumps(
                {
                    "quest_ready": _frame_summary(
                        first_frame,
                        gripper_enabled=args.enable_gripper,
                    )
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        if args.preview_only:
            return _preview(
                source,
                config,
                gripper_enabled=args.enable_gripper,
            )

        # Robot dependencies are unavailable and never imported in preview mode.
        import torch
        from polymetis import GripperInterface, RobotInterface

        robot = RobotInterface(ip_address=args.robot_ip)
        preflight = (
            wait_for_stationary_preflight(robot, timeout_sec=3.0)
            if args.non_interactive_ui
            else preflight_robot(robot)
        )
        print(json.dumps({"robot_ip": args.robot_ip, "preflight": preflight}, indent=2), flush=True)
        gripper_controller = None
        if args.enable_gripper:
            if not 1 <= args.gripper_port <= 65535:
                raise QuestTeleopError("gripper_port must be within [1, 65535]")
            gripper = GripperInterface(
                ip_address=args.robot_ip,
                port=args.gripper_port,
            )
            gripper_state = read_gripper(gripper)
            gripper_config = QuestGripperConfig(
                max_width_m=float(gripper_state["max_width_m"]),
                speed_m_s=args.gripper_speed_mm_s / 1000.0,
                force_n=args.gripper_force_n,
                command_hz=args.gripper_command_hz,
                max_closedness=args.gripper_max_closedness,
            )
            gripper_config.validate()
            gripper_controller = QuestGripperController(gripper, gripper_config)
            print(
                json.dumps(
                    {
                        "quest_gripper": {
                            **gripper_state,
                            "mapping": "rightTrig 0=open, 1=closed; commands only while RG held",
                            "speed_mm_s": args.gripper_speed_mm_s,
                            "force_n": args.gripper_force_n,
                            "max_closedness": args.gripper_max_closedness,
                            "min_width_mm": gripper_config.min_width_m * 1000.0,
                        }
                    },
                    indent=2,
                ),
                flush=True,
            )
        if args.check_only:
            print(
                "CHECK ONLY: no Polymetis control policy or gripper command was started.",
                flush=True,
            )
            return 0
        if not config.enforce_workspace_limits:
            print(
                "\nWARNING: SOFTWARE CARTESIAN WORKSPACE LIMIT IS DISABLED. "
                "Only deadman/rate/tracking/robot watchdogs remain.",
                flush=True,
            )
        if not args.non_interactive_ui and not _confirmation(
            gripper_enabled=args.enable_gripper
        ):
            print("Confirmation did not match; no control policy was started.", file=sys.stderr)
            return 2
        if args.non_interactive_ui:
            print(
                "UI CONTROL MODE: typed confirmation skipped; all runtime safety gates remain active.",
                flush=True,
            )

        # Demand a fresh post-confirmation frame and a released deadman.
        first_frame = source.wait_for_frame(args.quest_wait_sec)
        if bool(first_frame.buttons.get("RG", False)):
            print("Release right side-grip RG, then rerun; no policy was started.", file=sys.stderr)
            return 2
        if args.enable_gripper and right_trigger_closedness(first_frame.buttons) > 0.05:
            print(
                "Release the right index trigger, then rerun; no policy or gripper command "
                "was started.",
                file=sys.stderr,
            )
            return 2

        if args.telemetry_output is not None:
            telemetry_buffer = RobotTelemetryBuffer()
        last_print_second = -1
        last_deadman: Optional[bool] = None
        last_recovery: Optional[bool] = None
        last_workspace_clamped: Optional[bool] = None
        last_command_hold: Optional[bool] = None

        def status(tick: dict[str, object]) -> None:
            nonlocal last_print_second, last_deadman, last_recovery
            nonlocal last_workspace_clamped, last_command_hold
            if telemetry_buffer is not None:
                telemetry_buffer.append_tick(tick)
            second = int(float(tick["elapsed_sec"]))
            deadman = bool(tick["deadman_pressed"])
            recovery = bool(tick["workspace_recovery_active"])
            workspace_clamped = bool(tick["workspace_clamped"])
            command_hold = bool(tick["command_hold_latched"])
            if (
                second != last_print_second
                or deadman != last_deadman
                or recovery != last_recovery
                or workspace_clamped != last_workspace_clamped
                or command_hold != last_command_hold
            ):
                last_print_second = second
                last_deadman = deadman
                last_recovery = recovery
                last_workspace_clamped = workspace_clamped
                last_command_hold = command_hold
                if command_hold:
                    state = "SAFETY HOLD (release RG before resuming)"
                else:
                    state = "MOVE ENABLED (RG held)" if deadman else "HOLD (RG released)"
                gripper_tick = tick.get("gripper")
                gripper_text = ""
                if isinstance(gripper_tick, dict):
                    target_width_mm = gripper_tick.get("target_width_mm")
                    width_text = (
                        "none"
                        if target_width_mm is None
                        else f"{float(target_width_mm):.1f}mm"
                    )
                    gripper_text = (
                        f"; trigger={float(gripper_tick['closedness']):.2f}; "
                        f"gripper_closedness="
                        f"{float(gripper_tick['commanded_closedness']):.2f}; "
                        f"gripper_target={width_text}"
                    )
                print(
                    f"[quest] {state}; elapsed={second}s; "
                    f"target={tick['target_position_m']}; "
                    f"workspace_clamped={workspace_clamped}; "
                    f"recovery_active={recovery}; "
                    f"command_failure_streak={tick['command_failure_streak']}"
                    f"{gripper_text}",
                    flush=True,
                )

        print(
            "\nTELEOP STARTING IN HOLD MODE. Right controls:\n"
            "  hold RG side-grip = allow motion; release RG = immediate hold\n"
            "  click RJ while RG released = re-zero controller forward direction\n"
            "  press B = stop session\n"
            + (
                "  while RG is held: right index trigger 0=open, 1=closed "
                f"(limited to {args.gripper_max_closedness:.2f})\n"
                if args.enable_gripper
                else "  index trigger/gripper = disabled\n"
            ),
            flush=True,
        )
        result = run_right_quest_teleop(
            robot,
            frame_source=source,
            config=config,
            tensor_factory=lambda values: torch.tensor(values, dtype=torch.float32),
            gripper_controller=gripper_controller,
            on_tick=status,
            preflight_timeout_sec=3.0 if args.non_interactive_ui else 0.0,
        )
        print(f"Teleoperation ended safely: {result}. Owned policy terminated.", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nCtrl+C: session stopped; any policy owned by this tool was terminated.", file=sys.stderr)
        return 130
    except (
        FreedriveError,
        GripperControlError,
        QuestTeleopError,
        TimeoutError,
        RuntimeError,
    ) as exc:
        print(f"TELEOP REFUSED/STOPPED: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None:
            source.close()
        if telemetry_buffer is not None:
            assert args.telemetry_output is not None
            telemetry_buffer.write_npz_atomic(args.telemetry_output)
            print(
                f"Saved real robot telemetry atomically: {args.telemetry_output} "
                f"({len(telemetry_buffer)} samples)",
                flush=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
