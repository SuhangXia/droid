from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fabric_droid.robot.freedrive_home import (
    FreedriveError,
    HomeSnapshot,
    LowImpedanceConfig,
    preflight_robot,
    run_low_impedance_freedrive,
    save_home_snapshot,
    wait_for_stationary_preflight,
)


class FakeRobot:
    def __init__(self, *, running: bool = False, error_code: int = 0, velocity: float = 0.0) -> None:
        self.running = running
        self.error_code = error_code
        self.velocity = velocity
        self.started = 0
        self.updated = 0
        self.terminated = 0

    def is_running_policy(self) -> bool:
        return self.running

    def get_robot_state(self) -> SimpleNamespace:
        return SimpleNamespace(
            joint_positions=[0.1, -0.2, 0.3, -1.2, 0.5, 1.8, 0.7],
            joint_velocities=[self.velocity] * 7,
            error_code=self.error_code,
            prev_command_successful=True,
            timestamp=SimpleNamespace(seconds=12, nanos=34),
        )

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray([0.45, -0.1, 0.5]), np.asarray([0.0, 0.0, 0.0, 1.0])

    def start_cartesian_impedance(self, *, Kx: np.ndarray, Kxd: np.ndarray) -> None:
        assert Kx.shape == (6,)
        assert Kxd.shape == (6,)
        self.started += 1
        self.running = True

    def update_desired_ee_pose(self, position: np.ndarray, quat: np.ndarray) -> int:
        assert position.shape == (3,)
        assert quat.shape == (4,)
        self.updated += 1
        return self.updated

    def terminate_current_policy(self) -> None:
        self.terminated += 1
        self.running = False


def test_low_impedance_config_rejects_high_stiffness() -> None:
    LowImpedanceConfig().validate()
    with pytest.raises(FreedriveError, match="non-low"):
        LowImpedanceConfig(stiffness=(51, 20, 20, 2, 2, 2)).validate()


def test_preflight_refuses_active_policy_and_motion() -> None:
    with pytest.raises(FreedriveError, match="already running"):
        preflight_robot(FakeRobot(running=True))
    with pytest.raises(FreedriveError, match="not stationary"):
        preflight_robot(FakeRobot(velocity=0.1))
    with pytest.raises(FreedriveError, match="error_code"):
        preflight_robot(FakeRobot(error_code=3))


def test_stationary_preflight_retries_only_residual_motion() -> None:
    robot = FakeRobot(velocity=0.0516)
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(duration: float) -> None:
        sleeps.append(duration)
        clock[0] += duration
        robot.velocity = 0.0

    result = wait_for_stationary_preflight(
        robot,
        timeout_sec=3.0,
        monotonic=lambda: clock[0],
        sleep=sleep,
    )
    assert result["max_abs_joint_velocity_rad_s"] == 0.0
    assert sleeps == [pytest.approx(0.1)]

    active = FakeRobot(running=True)
    with pytest.raises(FreedriveError, match="already running"):
        wait_for_stationary_preflight(
            active,
            timeout_sec=3.0,
            monotonic=lambda: clock[0],
            sleep=lambda _: pytest.fail("active policy must fail without retry"),
        )
    faulted = FakeRobot(error_code=3, velocity=0.0516)
    with pytest.raises(FreedriveError, match="error_code"):
        wait_for_stationary_preflight(
            faulted,
            timeout_sec=3.0,
            monotonic=lambda: clock[0],
            sleep=lambda _: pytest.fail("robot faults must fail without retry"),
        )


def test_freedrive_starts_follows_captures_and_terminates() -> None:
    robot = FakeRobot()
    snapshot = run_low_impedance_freedrive(
        robot,
        config=LowImpedanceConfig(),
        should_save=lambda: True,
        tensor_factory=lambda values: np.asarray(values, dtype=np.float32),
        sleep=lambda _: None,
    )
    assert snapshot is not None
    assert snapshot.joint_positions_rad[0] == pytest.approx(0.1)
    assert robot.started == 1
    assert robot.updated == 1
    assert robot.terminated == 1
    assert not robot.running


def test_freedrive_library_gate_refuses_active_policy() -> None:
    robot = FakeRobot(running=True)
    with pytest.raises(FreedriveError, match="already running"):
        run_low_impedance_freedrive(
            robot,
            config=LowImpedanceConfig(),
            should_save=lambda: True,
            tensor_factory=lambda values: np.asarray(values, dtype=np.float32),
            sleep=lambda _: None,
        )
    assert robot.started == 0
    assert robot.terminated == 0


def test_home_snapshot_is_atomic_template_compatible_and_guarded(tmp_path: Path) -> None:
    path = tmp_path / "robot" / "home.json"
    snapshot = HomeSnapshot(
        position_m=(0.4, -0.1, 0.5),
        quat_xyzw=(0.0, 0.0, 0.0, 2.0),
        joint_positions_rad=(0.1, -0.2, 0.3, -1.2, 0.5, 1.8, 0.7),
        robot_timestamp_seconds=12,
        robot_timestamp_nanos=34,
    )
    save_home_snapshot(
        path,
        snapshot,
        robot_ip="192.168.0.116",
        source="test",
        config=LowImpedanceConfig(),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "franka_home_pose"
    assert payload["frame"] == "franka_base"
    assert payload["body"] == "flange"
    assert payload["quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert len(payload["T_base_flange"]) == 4
    assert len(payload["joint_positions_rad"]) == 7
    assert not list(path.parent.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        save_home_snapshot(
            path,
            snapshot,
            robot_ip="192.168.0.116",
            source="test",
            config=LowImpedanceConfig(),
        )
