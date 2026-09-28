from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fabric_droid.robot.go_home import (
    GoHomeError,
    SavedHome,
    load_saved_home,
    run_go_home,
)


class FakeHomeRobot:
    def __init__(self, *, running: bool = False) -> None:
        self.running = running
        self.started = 0
        self.terminated = 0
        self.polls_after_start = 0
        self.current = np.zeros(7)
        self.target = np.zeros(7)

    def is_running_policy(self) -> bool:
        if self.running and self.started:
            self.polls_after_start += 1
            if self.polls_after_start >= 3:
                self.running = False
                self.current = self.target.copy()
        return self.running

    def get_robot_state(self) -> SimpleNamespace:
        return SimpleNamespace(
            joint_positions=self.current.tolist(),
            joint_velocities=[0.0] * 7,
            error_code=0,
            prev_command_successful=True,
            timestamp=SimpleNamespace(seconds=1, nanos=2),
        )

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray([0.4, 0.0, 0.5]), np.asarray([0.0, 0.0, 0.0, 1.0])

    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        *,
        time_to_go: float,
        blocking: bool,
    ) -> None:
        assert time_to_go >= 3.0
        assert blocking is False
        self.target = np.asarray(positions)
        self.started += 1
        self.running = True

    def terminate_current_policy(self, return_log: bool = False) -> None:
        assert return_log is False
        self.terminated += 1
        self.running = False


def saved_home(tmp_path: Path, *, robot_ip: str = "192.168.0.116") -> Path:
    path = tmp_path / "home.json"
    path.write_text(
        json.dumps(
            {
                "name": "franka_home_pose",
                "frame": "franka_base",
                "robot_ip": robot_ip,
                "joint_positions_rad": [0.1] * 7,
                "position": [0.4, 0.0, 0.5],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_home_validates_robot_identity(tmp_path: Path) -> None:
    home = load_saved_home(saved_home(tmp_path), expected_robot_ip="192.168.0.116")
    assert home.joint_positions_rad == pytest.approx([0.1] * 7)
    with pytest.raises(GoHomeError, match="captured"):
        load_saved_home(saved_home(tmp_path, robot_ip="10.0.0.1"), expected_robot_ip="192.168.0.116")


def test_active_unknown_policy_is_refused_and_not_terminated(tmp_path: Path) -> None:
    robot = FakeHomeRobot(running=True)
    home = load_saved_home(saved_home(tmp_path), expected_robot_ip="192.168.0.116")
    with pytest.raises(Exception, match="already running"):
        run_go_home(
            robot,
            home,
            time_to_go_sec=10.0,
            tensor_factory=lambda values: np.asarray(values),
            sleep=lambda _: None,
        )
    assert robot.started == 0
    assert robot.terminated == 0


def test_home_motion_reaches_saved_joints(tmp_path: Path) -> None:
    robot = FakeHomeRobot()
    home = load_saved_home(saved_home(tmp_path), expected_robot_ip="192.168.0.116")
    result = run_go_home(
        robot,
        home,
        time_to_go_sec=10.0,
        tensor_factory=lambda values: np.asarray(values),
        sleep=lambda _: None,
    )
    assert result["result"] == "home_reached"
    assert robot.started == 1
    assert robot.terminated == 0
    assert robot.current == pytest.approx([0.1] * 7)


def test_already_home_does_not_start_policy(tmp_path: Path) -> None:
    robot = FakeHomeRobot()
    robot.current[:] = 0.1
    home = load_saved_home(saved_home(tmp_path), expected_robot_ip="192.168.0.116")
    result = run_go_home(
        robot,
        home,
        time_to_go_sec=10.0,
        tensor_factory=lambda values: np.asarray(values),
    )
    assert result["result"] == "already_home"
    assert robot.started == 0

