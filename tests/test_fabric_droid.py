from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from fabric_droid.capture.droid_hook import CallbackLifecycleHook, ForceInterlockLifecycleHook
from fabric_droid.capture.safety import ForceSafetyGate, PreflightError
from fabric_droid.io_utils import atomic_write_json
from fabric_droid.schemas import (
    CANONICAL_TASK_INSTRUCTION,
    EpisodeMetadata,
    EventMarker,
    ForceCalibration,
)
from fabric_droid.sensors.ati import ATISample, ATIStream, parse_ati_payload
from fabric_droid.sensors.synthetic import SyntheticATIStream
from fabric_droid.sync.clock import analyze_timestamps


def metadata() -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_id="episode_test",
        task_instruction=CANONICAL_TASK_INSTRUCTION,
        destination_tray="target_tray",
        source_slot="slot",
        swatch_uid="swatch_001",
        split="train",
        start_pose_bucket="ready",
        success=True,
        failure_reason="",
        operator="test",
        session_id="session_test",
        robot_id="disabled",
        camera_serials={},
        gelsight_serial="g",
        ati_serial="a",
        calibration_id="uncalibrated",
        software_git_commits={"droid": "test"},
    )


def test_schema_and_event_contract() -> None:
    assert metadata().to_dict()["split"] == "train"
    event = EventMarker("probe_complete", time.monotonic_ns())
    assert event.to_dict()["source"] == "manual"
    with pytest.raises(ValueError):
        EventMarker("not_an_event", time.monotonic_ns())


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"1, 2, 3, 4, 5, 6", np.arange(1, 7)),
        ('{"wrench": {"fx": 1, "fy": 2, "fz": 3, "tx": 4, "ty": 5, "tz": 6}}', np.arange(1, 7)),
        ("[0, 1, 2, 3, 4, 5, 6]", np.arange(1, 7)),
    ],
)
def test_ati_parser(payload: bytes | str, expected: np.ndarray) -> None:
    values, _, status = parse_ati_payload(payload)
    np.testing.assert_allclose(values, expected)
    assert status == "ok"


def test_ati_readiness_rejects_all_zero_and_accepts_live_wrench() -> None:
    stream = ATIStream()
    now = time.monotonic_ns()
    samples = [
        ATISample(
            timestamp_monotonic_ns=now - (63 - index) * 2_000_000,
            timestamp_wall_ns=time.time_ns(),
            sample_index=index,
            fx=0.0,
            fy=0.0,
            fz=0.0,
            tx=0.0,
            ty=0.0,
            tz=0.0,
        )
        for index in range(64)
    ]
    with stream._lock:  # noqa: SLF001
        stream.samples.extend(samples)
        stream._total_sample_count = len(samples)  # noqa: SLF001
    report = stream.readiness()
    assert not report["ready"]
    assert "recent wrench samples are all zero" in report["reasons"]

    live = ATISample(
        timestamp_monotonic_ns=time.monotonic_ns(),
        timestamp_wall_ns=time.time_ns(),
        sample_index=64,
        fx=0.0,
        fy=0.0,
        fz=0.25,
        tx=0.0,
        ty=0.0,
        tz=0.0,
    )
    with stream._lock:  # noqa: SLF001
        stream.samples.append(live)
        stream._total_sample_count += 1  # noqa: SLF001
    assert stream.readiness()["ready"]
    assert not stream.readiness(
        since_monotonic_ns=live.timestamp_monotonic_ns + 1
    )["ready"]
    with stream._lock:  # noqa: SLF001
        stream._max_inter_sample_gap_ns = 1_100_000_000  # noqa: SLF001
    report = stream.readiness(max_gap_sec=1.0)
    assert not report["ready"]
    assert any("historical packet gap" in reason for reason in report["reasons"])


def test_ati_gap_monitor_counts_start_to_first_episode_packet() -> None:
    stream = ATIStream()
    start_ns = 10_000_000_000
    stream._append(  # noqa: SLF001
        np.ones(6),
        arrival_ns=start_ns - 1_000_000,
        wall_ns=time.time_ns(),
    )
    stream.reset_gap_monitor(start_ns)
    stream._append(  # noqa: SLF001
        np.ones(6),
        arrival_ns=start_ns + 1_100_000_000,
        wall_ns=time.time_ns(),
    )
    report = stream.readiness(max_gap_sec=1.0)
    assert report["max_gap_sec_observed"] == pytest.approx(1.1)
    assert any("historical packet gap" in reason for reason in report["reasons"])


def test_synthetic_ati_preserves_all_samples_and_monotonic_clock() -> None:
    stream = SyntheticATIStream(500.0)
    stream.start()
    time.sleep(0.25)
    stream.stop()
    assert len(stream.samples) >= 100
    assert [sample.sample_index for sample in stream.samples] == list(range(len(stream.samples)))
    report = analyze_timestamps([sample.timestamp_monotonic_ns for sample in stream.samples])
    assert report.monotonic
    assert 430 <= report.measured_hz <= 570


def test_force_gate_and_uncalibrated_normal_force() -> None:
    gate = ForceSafetyGate()
    assert gate.evaluate_raw([1, 2, 3, 0, 0, 0]).allow_further_closing
    assert gate.evaluate_raw([30, 0, 0, 0, 0, 0]).abort_episode
    with pytest.raises(PreflightError):
        gate.normal_force([0] * 6, ForceCalibration("missing"))


def test_atomic_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "value.json"
    atomic_write_json(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    assert not list(path.parent.glob("*.tmp"))


def test_lifecycle_hook_dependency_injection() -> None:
    calls: list[tuple[str, object]] = []
    hook = CallbackLifecycleHook(
        on_start=lambda value: calls.append(("start", value)),
        on_step=lambda value: calls.append(("step", value)),
        on_end=lambda timestamp, info: calls.append(("end", (timestamp, info))),
    )
    hook.on_episode_start(1)
    hook.on_timestep({"observation": {}, "action": {}})
    hook.on_episode_end(2, {"success": True})
    assert [call[0] for call in calls] == ["start", "step", "end"]


def test_force_interlock_runs_before_action() -> None:
    blocked = ForceInterlockLifecycleHook(
        latest_wrench=lambda: np.asarray([13.0, 0, 0, 0, 0, 0]),
        block_closing=lambda action: np.zeros_like(action),
        abort_episode=lambda reason: None,
    )
    np.testing.assert_array_equal(blocked.before_action({}, np.ones(8)), np.zeros(8))
    aborted: list[str] = []
    hard = ForceInterlockLifecycleHook(
        latest_wrench=lambda: np.asarray([30.0, 0, 0, 0, 0, 0]),
        block_closing=lambda action: action,
        abort_episode=aborted.append,
    )
    with pytest.raises(RuntimeError):
        hard.before_action({}, np.ones(8))
    assert aborted
