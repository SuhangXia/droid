from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from fabric_droid.robot.gripper_control import (
    MAX_SAFE_CLOSEDNESS,
    GripperControlError,
    GripperMotion,
    closedness_for_width,
    move_gripper,
    read_gripper,
    wait_for_gripper_width,
    width_for_closedness,
)


class FakeGripper:
    def __init__(
        self,
        *,
        width: float = 0.08,
        success: bool = True,
        error_code: int = 0,
        moving: bool = False,
    ) -> None:
        self.metadata = SimpleNamespace(max_width=0.0808, hz=30)
        self.width = width
        self.success = success
        self.error_code = error_code
        self.moving = moving
        self.commands: list[dict[str, float | bool]] = []

    def get_state(self) -> SimpleNamespace:
        return SimpleNamespace(
            width=self.width,
            is_grasped=False,
            is_moving=self.moving,
            prev_command_successful=self.success,
            error_code=self.error_code,
            timestamp=SimpleNamespace(seconds=1, nanos=2),
        )

    def goto(self, *, width: float, speed: float, force: float, blocking: bool) -> None:
        self.commands.append(
            {"width": width, "speed": speed, "force": force, "blocking": blocking}
        )
        self.width = width


def test_read_gripper_is_read_only_and_reports_mm() -> None:
    gripper = FakeGripper()
    state = read_gripper(gripper)
    assert state["width_mm"] == pytest.approx(80.0)
    assert state["max_width_mm"] == pytest.approx(80.8)
    assert gripper.commands == []


def test_motion_bounds_are_guarded() -> None:
    GripperMotion(width_m=0.04, speed_m_s=0.01, force_n=5.0).validate(max_width_m=0.0808)
    with pytest.raises(GripperControlError, match="outside"):
        GripperMotion(width_m=0.09).validate(max_width_m=0.0808)
    with pytest.raises(GripperControlError, match="speed"):
        GripperMotion(width_m=0.04, speed_m_s=0.1).validate(max_width_m=0.0808)
    with pytest.raises(GripperControlError, match="force"):
        GripperMotion(width_m=0.04, force_n=50.0).validate(max_width_m=0.0808)


def test_hard_97_percent_closedness_limit() -> None:
    max_width_m = 0.0808
    minimum_width_m = width_for_closedness(max_width_m, MAX_SAFE_CLOSEDNESS)
    GripperMotion(width_m=minimum_width_m).validate(max_width_m=max_width_m)
    assert closedness_for_width(max_width_m, minimum_width_m) == pytest.approx(0.98)
    with pytest.raises(GripperControlError, match="hard safety limit"):
        GripperMotion(width_m=minimum_width_m - 0.0001).validate(max_width_m=max_width_m)


def test_95_percent_target_is_above_hard_limit() -> None:
    max_width_m = 0.0808
    target = width_for_closedness(max_width_m, 0.95)
    assert target == pytest.approx(0.00404)
    GripperMotion(width_m=target).validate(max_width_m=max_width_m)


def test_move_gripper_sends_one_blocking_command_and_reads_result() -> None:
    gripper = FakeGripper()
    result = move_gripper(
        gripper,
        GripperMotion(width_m=0.07, speed_m_s=0.01, force_n=5.0),
    )
    assert len(gripper.commands) == 1
    assert gripper.commands[0] == {
        "width": 0.07,
        "speed": 0.01,
        "force": 5.0,
        "blocking": True,
    }
    assert result["after"]["width_mm"] == pytest.approx(70.0)


def test_verified_move_waits_for_measured_target() -> None:
    gripper = FakeGripper()
    result = move_gripper(
        gripper,
        GripperMotion(width_m=0.00404, speed_m_s=0.01, force_n=5.0),
        verify_width=True,
    )
    assert result["after"]["width_mm"] == pytest.approx(4.04)


def test_wait_for_width_times_out_if_state_does_not_converge(monkeypatch: pytest.MonkeyPatch) -> None:
    gripper = FakeGripper(width=0.08)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr("fabric_droid.robot.gripper_control.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("fabric_droid.robot.gripper_control.time.sleep", lambda _: None)
    with pytest.raises(GripperControlError, match="did not reach"):
        wait_for_gripper_width(
            gripper,
            target_width_m=0.004,
            timeout_s=0.5,
        )


def test_wait_for_width_does_not_accept_a_still_moving_gripper(monkeypatch: pytest.MonkeyPatch) -> None:
    gripper = FakeGripper(width=0.004, moving=True)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr("fabric_droid.robot.gripper_control.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr("fabric_droid.robot.gripper_control.time.sleep", lambda _: None)
    with pytest.raises(GripperControlError, match="did not reach"):
        wait_for_gripper_width(
            gripper,
            target_width_m=0.004,
            timeout_s=0.5,
        )


def test_failed_preflight_never_commands_gripper() -> None:
    gripper = FakeGripper(success=False)
    with pytest.raises(GripperControlError, match="previous"):
        move_gripper(gripper, GripperMotion(width_m=0.07))
    assert gripper.commands == []


def test_ui_mode_skips_only_typed_gripper_confirmation() -> None:
    import tools.franka_gripper_control as cli

    source = inspect.getsource(cli.main)
    assert "not args.non_interactive_ui and not _confirmation" in source
    assert "args.enable_gripper" in source
    assert "args.preflight_confirmed" in source
    assert "after_width_m < max_width_m - 0.003" in source
    assert "verify_width=True" in source
