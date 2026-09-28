from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from fabric_droid.robot.quest_teleop import (
    QuestFrame,
    QuestGripperConfig,
    QuestGripperController,
    QuestTeleopConfig,
    QuestTeleopError,
    RightQuestTargetMapper,
    apply_workspace_recovery_guard,
    right_trigger_closedness,
    run_right_quest_teleop,
    validate_quest_frame,
)


def frame(
    *,
    sequence: int = 1,
    timestamp_ns: int = 1_000_000_000,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rg: bool = False,
    rj: bool = False,
    b: bool = False,
    trigger: float = 0.0,
) -> QuestFrame:
    pose = np.eye(4)
    pose[:3, 3] = position
    return QuestFrame(
        pose=pose,
        buttons={"RG": rg, "RJ": rj, "B": b, "rightTrig": (trigger,)},
        received_monotonic_ns=timestamp_ns,
        sequence=sequence,
    )


class FakeSource:
    def __init__(self, frames: list[QuestFrame]) -> None:
        self.frames = frames
        self.index = 0

    def latest_frame(self) -> QuestFrame:
        result = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return result


class FakeRobot:
    def __init__(
        self,
        *,
        running: bool = False,
        running_command_results: list[bool] | None = None,
    ) -> None:
        self.running = running
        self.running_command_results = list(running_command_results or [])
        self.started = 0
        self.terminated = 0
        self.updated: list[tuple[np.ndarray, np.ndarray]] = []
        self.position = np.asarray([0.45, 0.0, 0.45])
        self.quat = np.asarray([0.0, 0.0, 0.0, 1.0])

    def is_running_policy(self) -> bool:
        return self.running

    def get_robot_state(self) -> SimpleNamespace:
        command_successful = True
        if self.running and self.running_command_results:
            command_successful = self.running_command_results.pop(0)
        return SimpleNamespace(
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            error_code=0,
            prev_command_successful=command_successful,
            timestamp=SimpleNamespace(seconds=1, nanos=2),
        )

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.position.copy(), self.quat.copy()

    def start_cartesian_impedance(self, *, Kx: np.ndarray, Kxd: np.ndarray) -> None:
        assert Kx.shape == (6,)
        assert Kxd.shape == (6,)
        self.started += 1
        self.running = True

    def update_desired_ee_pose(self, position: np.ndarray, quat: np.ndarray) -> int:
        self.updated.append((position.copy(), quat.copy()))
        return len(self.updated)

    def terminate_current_policy(self, return_log: bool = False) -> None:
        assert return_log is False
        self.terminated += 1
        self.running = False


class FakeQuestGripper:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(max_width=0.08, hz=30)
        self.width = 0.08
        self.commands: list[dict[str, float | bool]] = []

    def get_state(self) -> SimpleNamespace:
        return SimpleNamespace(
            width=self.width,
            is_grasped=False,
            is_moving=False,
            prev_command_successful=True,
            error_code=0,
            timestamp=SimpleNamespace(seconds=1, nanos=2),
        )

    def goto(self, *, width: float, speed: float, force: float, blocking: bool) -> None:
        self.commands.append(
            {"width": width, "speed": speed, "force": force, "blocking": blocking}
        )
        self.width = width


def test_watchdog_rejects_stale_frame() -> None:
    with pytest.raises(QuestTeleopError, match="watchdog expired"):
        validate_quest_frame(frame(timestamp_ns=0), now_ns=1_000_000_000, timeout_sec=0.25)


def test_right_trigger_matches_droid_closedness_convention() -> None:
    assert right_trigger_closedness(frame(trigger=0.0).buttons) == pytest.approx(0.0)
    assert right_trigger_closedness(frame(trigger=1.0).buttons) == pytest.approx(1.0)
    with pytest.raises(QuestTeleopError, match="rightTrig"):
        right_trigger_closedness({"RG": False})


def test_gripper_only_commands_while_deadman_and_holds_on_release() -> None:
    gripper = FakeQuestGripper()
    controller = QuestGripperController(
        gripper,
        QuestGripperConfig(
            max_width_m=0.08,
            speed_m_s=0.02,
            force_n=10.0,
            command_hz=15.0,
        ),
    )
    disabled = controller.step(
        frame(sequence=1, trigger=0.5),
        movement_enabled=False,
        now_ns=1_000_000_000,
    )
    assert not disabled["command_sent"]
    assert gripper.commands == []

    enabled = controller.step(
        frame(sequence=2, rg=True, trigger=0.5),
        movement_enabled=True,
        now_ns=1_100_000_000,
    )
    assert enabled["command_sent"]
    assert gripper.commands[-1]["width"] == pytest.approx(0.04)
    assert gripper.commands[-1]["blocking"] is False

    held = controller.step(
        frame(sequence=3, trigger=1.0),
        movement_enabled=False,
        now_ns=1_200_000_000,
    )
    assert held["command_sent"]
    assert held["hold_reason"] == "deadman_or_session_stop"
    assert gripper.commands[-1]["width"] == pytest.approx(0.04)


def test_gripper_closure_is_capped_at_97_percent() -> None:
    gripper = FakeQuestGripper()
    controller = QuestGripperController(
        gripper,
        QuestGripperConfig(
            max_width_m=0.08,
            speed_m_s=0.02,
            force_n=10.0,
            max_closedness=0.97,
        ),
    )
    status = controller.step(
        frame(sequence=1, rg=True, trigger=1.0),
        movement_enabled=True,
        now_ns=1_000_000_000,
    )
    assert status["closedness"] == pytest.approx(1.0)
    assert status["commanded_closedness"] == pytest.approx(0.97)
    assert status["target_width_m"] == pytest.approx(0.0024)
    assert gripper.commands[-1]["width"] == pytest.approx(0.0024)


def test_gripper_default_closure_is_capped_at_98_percent() -> None:
    config = QuestGripperConfig(max_width_m=0.0808)
    assert config.max_closedness == pytest.approx(0.98)
    assert config.min_width_m == pytest.approx(0.001616)


def test_gripper_safety_hold_never_commands_below_minimum_width() -> None:
    gripper = FakeQuestGripper()
    controller = QuestGripperController(
        gripper,
        QuestGripperConfig(max_width_m=0.08, max_closedness=0.97),
    )
    controller.step(
        frame(sequence=1, rg=True, trigger=1.0),
        movement_enabled=True,
        now_ns=1_000_000_000,
    )
    # Simulate compliant overshoot below the configured minimum width.
    gripper.width = 0.0
    held = controller.step(
        frame(sequence=2, trigger=1.0),
        movement_enabled=False,
        now_ns=1_100_000_000,
    )
    assert held["target_width_m"] == pytest.approx(0.0024)
    assert gripper.commands[-1]["width"] == pytest.approx(0.0024)


def test_zero_duration_is_unlimited_but_negative_is_rejected() -> None:
    QuestTeleopConfig(max_duration_sec=0.0).validate()
    with pytest.raises(QuestTeleopError, match="non-negative"):
        QuestTeleopConfig(max_duration_sec=-1.0).validate()


def test_mapper_requires_released_deadman_to_arm() -> None:
    mapper = RightQuestTargetMapper(QuestTeleopConfig())
    with pytest.raises(QuestTeleopError, match="release"):
        mapper.arm(frame(rg=True), [0.45, 0.0, 0.45], [0.0, 0.0, 0.0, 1.0])


def test_deadman_release_holds_measured_pose_and_b_exits() -> None:
    mapper = RightQuestTargetMapper(QuestTeleopConfig())
    mapper.arm(frame(), [0.45, 0.0, 0.45], [0.0, 0.0, 0.0, 1.0])
    first = mapper.step(frame(sequence=2, rg=True), [0.45, 0.0, 0.45], [0, 0, 0, 1])
    assert first.send_target and first.deadman_pressed
    released = mapper.step(frame(sequence=3), [0.46, 0.01, 0.44], [0, 0, 0, 1])
    assert released.send_target and not released.deadman_pressed
    assert released.position_m == pytest.approx([0.46, 0.01, 0.44])
    held = mapper.step(frame(sequence=4), [0.47, 0.02, 0.43], [0, 0, 0, 1])
    assert not held.send_target
    assert held.position_m == pytest.approx([0.46, 0.01, 0.44])
    assert mapper.step(frame(sequence=5, b=True), [0.47, 0.02, 0.43], [0, 0, 0, 1]).exit_requested


def test_motion_is_rate_limited_and_workspace_clamped() -> None:
    config = QuestTeleopConfig(
        control_hz=10.0,
        max_translation_speed_m_s=0.05,
        workspace_min_m=(0.44, -0.01, 0.44),
        workspace_max_m=(0.46, 0.01, 0.46),
    )
    mapper = RightQuestTargetMapper(config)
    mapper.arm(frame(), [0.45, 0.0, 0.45], [0, 0, 0, 1])
    mapper.step(frame(sequence=2, rg=True), [0.45, 0.0, 0.45], [0, 0, 0, 1])
    decision = mapper.step(
        frame(sequence=3, rg=True, position=(0.0, -1.0, 0.0)),
        [0.45, 0.0, 0.45],
        [0, 0, 0, 1],
    )
    assert np.linalg.norm(decision.position_m - [0.45, 0.0, 0.45]) <= 0.005001
    assert decision.position_m[0] <= 0.46


def test_recovery_blocks_outward_motion_and_rate_limits_inward_motion() -> None:
    config = QuestTeleopConfig(
        control_hz=10.0,
        max_translation_speed_m_s=0.05,
        workspace_min_m=(0.45, -0.12, 0.45),
        workspace_max_m=(0.63, 0.12, 0.70),
        workspace_recovery_margin_m=0.02,
    )
    mapper = RightQuestTargetMapper(config)
    initial = [0.635, 0.0, 0.55]
    mapper.arm(frame(), initial, [0, 0, 0, 1])
    enabled = mapper.step(frame(sequence=2, rg=True), initial, [0, 0, 0, 1])
    assert enabled.workspace_recovery_active

    # QUEST_TO_ROBOT maps negative controller Y to positive robot X: outward.
    outward = mapper.step(
        frame(sequence=3, rg=True, position=(0.0, -0.02, 0.0)),
        initial,
        [0, 0, 0, 1],
    )
    assert outward.position_m[0] == pytest.approx(0.635)
    assert outward.workspace_clamped
    assert outward.workspace_recovery_active

    # Positive controller Y maps to negative robot X: inward, at <= 5 mm/tick.
    inward = mapper.step(
        frame(sequence=4, rg=True, position=(0.0, 0.02, 0.0)),
        initial,
        [0, 0, 0, 1],
    )
    assert 0.63 <= inward.position_m[0] < 0.635
    assert 0.635 - inward.position_m[0] <= 0.005001
    assert inward.workspace_recovery_active


def test_recovery_guard_refuses_target_beyond_margin() -> None:
    with pytest.raises(QuestTeleopError, match="recovery margin"):
        apply_workspace_recovery_guard(
            [0.66, 0.0, 0.55],
            [0.66, 0.0, 0.55],
            workspace_min_m=[0.45, -0.12, 0.45],
            workspace_max_m=[0.63, 0.12, 0.70],
            recovery_margin_m=0.02,
        )


def test_mapper_refuses_measured_pose_beyond_recovery_margin() -> None:
    config = QuestTeleopConfig(
        workspace_min_m=(0.45, -0.12, 0.45),
        workspace_max_m=(0.63, 0.12, 0.70),
        workspace_recovery_margin_m=0.02,
    )
    mapper = RightQuestTargetMapper(config)
    mapper.arm(frame(), [0.635, 0.0, 0.55], [0, 0, 0, 1])
    with pytest.raises(QuestTeleopError, match="measured flange"):
        mapper.step(frame(sequence=2), [0.651, 0.0, 0.55], [0, 0, 0, 1])


def test_disabled_workspace_allows_unbounded_target_without_clamping() -> None:
    config = QuestTeleopConfig(
        control_hz=10.0,
        max_translation_speed_m_s=0.05,
        enforce_workspace_limits=False,
        workspace_min_m=(0.45, -0.12, 0.45),
        workspace_max_m=(0.63, 0.12, 0.70),
    )
    mapper = RightQuestTargetMapper(config)
    initial = [1.50, -1.0, 1.20]
    mapper.arm(frame(), initial, [0, 0, 0, 1])
    mapper.step(frame(sequence=2, rg=True), initial, [0, 0, 0, 1])
    decision = mapper.step(
        frame(sequence=3, rg=True, position=(0.0, -0.02, 0.0)),
        initial,
        [0, 0, 0, 1],
    )
    assert decision.position_m[0] > 1.50
    assert not decision.workspace_clamped
    assert not decision.workspace_recovery_active


def test_active_unknown_policy_is_never_terminated() -> None:
    robot = FakeRobot(running=True)
    with pytest.raises(Exception, match="already running"):
        run_right_quest_teleop(
            robot,
            frame_source=FakeSource([frame()]),
            config=QuestTeleopConfig(max_duration_sec=1.0),
            tensor_factory=lambda values: np.asarray(values),
            monotonic_ns=lambda: 1_000_000_000,
            sleep=lambda _: None,
        )
    assert robot.started == 0
    assert robot.terminated == 0


def test_b_stops_and_owned_policy_is_terminated() -> None:
    robot = FakeRobot()
    frames = [frame(sequence=1), frame(sequence=2, b=True)]
    result = run_right_quest_teleop(
        robot,
        frame_source=FakeSource(frames),
        config=QuestTeleopConfig(max_duration_sec=1.0),
        tensor_factory=lambda values: np.asarray(values, dtype=np.float32),
        monotonic_ns=lambda: 1_000_000_000,
        sleep=lambda _: None,
    )
    assert result == "b_button"
    assert robot.started == 1
    assert robot.terminated == 1
    assert len(robot.updated) == 1


def test_unlimited_session_can_still_exit_with_b() -> None:
    robot = FakeRobot()
    frames = [frame(sequence=1), frame(sequence=2, b=True)]
    result = run_right_quest_teleop(
        robot,
        frame_source=FakeSource(frames),
        config=QuestTeleopConfig(max_duration_sec=0.0),
        tensor_factory=lambda values: np.asarray(values, dtype=np.float32),
        monotonic_ns=lambda: 1_000_000_000,
        sleep=lambda _: None,
    )
    assert result == "b_button"
    assert robot.terminated == 1


def test_stale_frame_after_start_terminates_only_owned_policy() -> None:
    robot = FakeRobot()
    frames = [frame(sequence=1), frame(sequence=2, timestamp_ns=0)]
    with pytest.raises(QuestTeleopError, match="watchdog expired"):
        run_right_quest_teleop(
            robot,
            frame_source=FakeSource(frames),
            config=QuestTeleopConfig(max_duration_sec=1.0),
            tensor_factory=lambda values: np.asarray(values, dtype=np.float32),
            monotonic_ns=lambda: 1_000_000_000,
            sleep=lambda _: None,
        )
    assert robot.started == 1
    assert robot.terminated == 1


def test_single_command_failure_enters_hold_then_recovers() -> None:
    robot = FakeRobot(running_command_results=[False, True])
    frames = [
        frame(sequence=1),
        frame(sequence=2, rg=True),
        frame(sequence=3),
        frame(sequence=4, b=True),
    ]
    ticks: list[dict[str, object]] = []
    result = run_right_quest_teleop(
        robot,
        frame_source=FakeSource(frames),
        config=QuestTeleopConfig(max_duration_sec=0.0),
        tensor_factory=lambda values: np.asarray(values, dtype=np.float32),
        monotonic_ns=lambda: 1_000_000_000,
        sleep=lambda _: None,
        on_tick=ticks.append,
    )
    assert result == "b_button"
    assert robot.terminated == 1
    assert ticks[0]["command_failure_streak"] == 1
    assert ticks[0]["command_hold_latched"] is True
    assert ticks[0]["deadman_pressed"] is False
    assert ticks[1]["command_failure_streak"] == 0
    assert ticks[1]["command_hold_latched"] is False


def test_consecutive_command_failures_stop_owned_policy() -> None:
    robot = FakeRobot(running_command_results=[False, False, False])
    frames = [
        frame(sequence=1),
        frame(sequence=2),
        frame(sequence=3),
        frame(sequence=4),
    ]
    with pytest.raises(QuestTeleopError, match="3 consecutive"):
        run_right_quest_teleop(
            robot,
            frame_source=FakeSource(frames),
            config=QuestTeleopConfig(max_duration_sec=0.0),
            tensor_factory=lambda values: np.asarray(values, dtype=np.float32),
            monotonic_ns=lambda: 1_000_000_000,
            sleep=lambda _: None,
        )
    assert robot.started == 1
    assert robot.terminated == 1


def test_ui_mode_skips_only_typed_confirmation() -> None:
    import tools.franka_quest_teleop as cli

    source = inspect.getsource(cli.main)
    assert "not args.non_interactive_ui and not _confirmation" in source
    assert "Release right side-grip RG" in source
    assert "right index trigger" in source
    assert "preflight_timeout_sec=3.0 if args.non_interactive_ui else 0.0" in source
    assert "RobotTelemetryBuffer" in source
    assert "write_npz_atomic(args.telemetry_output)" in source
