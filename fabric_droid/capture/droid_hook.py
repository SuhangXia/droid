"""DROID collector hook contract and dependency-injection adapter."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from fabric_droid.capture.recorder import FabricEpisodeRecorder, RecorderOptions
from fabric_droid.capture.safety import ForceSafetyGate
from fabric_droid.schemas import EpisodeMetadata, ForceCalibration


class CallbackLifecycleHook:
    """Adapt sidecar callbacks to DROID without importing sensor code in DROID.

    A hook factory is passed to ``DataCollecter(..., lifecycle_hook_factory=...)``.
    The factory receives the episode directory and DROID metadata. Existing
    callers see no behavior change because the factory defaults to ``None``.
    """

    def __init__(
        self,
        *,
        on_start: Callable[[int], None] | None = None,
        before_action: Callable[[dict[str, Any], Any], Any] | None = None,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        on_end: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> None:
        self._on_start = on_start
        self._before_action = before_action
        self._on_step = on_step
        self._on_end = on_end

    def on_episode_start(self, timestamp_monotonic_ns: int) -> None:
        if self._on_start is not None:
            self._on_start(timestamp_monotonic_ns)

    def on_timestep(self, timestep: dict[str, Any]) -> None:
        if self._on_step is not None:
            self._on_step(timestep)

    def before_action(self, observation: dict[str, Any], action: Any) -> Any:
        return action if self._before_action is None else self._before_action(observation, action)

    def on_episode_end(self, timestamp_monotonic_ns: int, controller_info: dict[str, Any]) -> None:
        if self._on_end is not None:
            self._on_end(timestamp_monotonic_ns, controller_info)


class ForceInterlockLifecycleHook(CallbackLifecycleHook):
    """Apply an ATI hardware interlock before DROID sends each action.

    Closing semantics are injected because different controllers encode the
    gripper command differently. On warning, ``block_closing`` must return a
    safe action (normally with the gripper command held). A hard stop invokes
    the robot owner's abort callback and terminates the episode.
    """

    def __init__(
        self,
        latest_wrench: Callable[[], np.ndarray | None],
        block_closing: Callable[[Any], Any],
        abort_episode: Callable[[str], None],
        gate: ForceSafetyGate | None = None,
    ) -> None:
        self.latest_wrench = latest_wrench
        self.block_closing = block_closing
        self.abort_episode = abort_episode
        self.gate = gate or ForceSafetyGate()
        super().__init__(before_action=self._check)

    def _check(self, observation: dict[str, Any], action: Any) -> Any:
        del observation
        wrench = self.latest_wrench()
        if wrench is None:
            self.abort_episode("ATI stream is unavailable")
            raise RuntimeError("ATI interlock aborted episode: no current wrench")
        decision = self.gate.evaluate_raw(wrench)
        if decision.abort_episode:
            self.abort_episode(decision.reason)
            raise RuntimeError(f"ATI interlock aborted episode: {decision.reason}")
        if not decision.allow_further_closing:
            return self.block_closing(action)
        return action


class DroidSidecarLifecycle:
    """Capture sensors beside the DROID-owned trajectory with atomic tactile close."""

    def __init__(
        self,
        episode_dir: Path,
        options: RecorderOptions,
        metadata: EpisodeMetadata,
        *,
        block_closing: Callable[[Any], Any] | None = None,
        abort_episode: Callable[[str], None] | None = None,
        safety_gate: ForceSafetyGate | None = None,
        calibration: ForceCalibration | None = None,
    ) -> None:
        self.episode_dir = episode_dir
        self.staging_root = episode_dir.parent / ".fabric_sidecar_staging"
        staged_options = replace(
            options,
            output_dir=self.staging_root,
            dry_run=False,
            robot_disabled=True,
            record_only=True,
        )
        self.recorder = FabricEpisodeRecorder(staged_options, metadata, calibration)
        self.block_closing = block_closing
        self.abort_episode = abort_episode
        self.safety_gate = safety_gate or ForceSafetyGate()

    def on_episode_start(self, timestamp_monotonic_ns: int) -> None:
        del timestamp_monotonic_ns
        self.recorder.start()

    def before_action(self, observation: dict[str, Any], action: Any) -> Any:
        del observation
        ati_stream = next((stream for stream in self.recorder.streams if hasattr(stream, "samples")), None)
        if ati_stream is None or not ati_stream.samples:
            reason = "ATI interlock has no current wrench"
            if self.abort_episode is not None:
                self.abort_episode(reason)
            raise RuntimeError(reason)
        sample = ati_stream.samples[-1]
        wrench = np.asarray([sample.fx, sample.fy, sample.fz, sample.tx, sample.ty, sample.tz])
        decision = self.safety_gate.evaluate_raw(wrench)
        if decision.abort_episode:
            if self.abort_episode is not None:
                self.abort_episode(decision.reason)
            raise RuntimeError(f"ATI interlock aborted episode: {decision.reason}")
        if not decision.allow_further_closing:
            if self.block_closing is None:
                raise RuntimeError("ATI warning threshold reached and no safe gripper-hold callback was supplied")
            return self.block_closing(action)
        return action

    def on_timestep(self, timestep: dict[str, Any]) -> None:
        del timestep

    def mark_event(self, name: str) -> None:
        self.recorder.mark_event(name, source="manual")

    @staticmethod
    def _move_new(source: Path, target: Path) -> None:
        if target.exists():
            raise FileExistsError(f"refusing to overwrite DROID episode file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)

    def on_episode_end(self, timestamp_monotonic_ns: int, controller_info: dict[str, Any]) -> None:
        del timestamp_monotonic_ns
        success = bool(controller_info.get("success", False))
        missing = {"probe_complete", "release_time"} - {event.name for event in self.recorder.events}
        reason = str(controller_info.get("failure_reason", ""))
        if missing:
            success = False
            reason = f"missing required manual events: {sorted(missing)}"
        staged_episode = self.recorder.close(success=success, failure_reason=reason)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        tactile_source = staged_episode / "tactile"
        if tactile_source.exists():
            self._move_new(tactile_source, self.episode_dir / "tactile")
        recordings = staged_episode / "recordings"
        if recordings.exists():
            for source in sorted(path for path in recordings.glob("**/*") if path.is_file()):
                self._move_new(source, self.episode_dir / "recordings" / source.relative_to(recordings))
        for source in staged_episode.glob("metadata_*.json"):
            self._move_new(
                source,
                self.episode_dir / f"fabric_{source.name}",
            )
        for name in ("preflight.json", "COMPLETE.json", "CAPTURE_INCOMPLETE.json"):
            source = staged_episode / name
            if source.exists():
                target_name = "fabric_sidecar_COMPLETE.json" if name == "COMPLETE.json" else name
                self._move_new(source, self.episode_dir / "tactile" / target_name)
        for directory in sorted(
            (path for path in staged_episode.glob("**/*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            staged_episode.rmdir()
            self.staging_root.rmdir()
        except OSError:
            pass


def make_droid_sidecar_hook_factory(
    options: RecorderOptions,
    metadata_builder: Callable[[Path, dict[str, Any]], EpisodeMetadata],
    *,
    block_closing: Callable[[Any], Any] | None = None,
    abort_episode: Callable[[str], None] | None = None,
    safety_gate: ForceSafetyGate | None = None,
    calibration: ForceCalibration | None = None,
) -> Callable[[str, dict[str, Any]], DroidSidecarLifecycle]:
    """Return the factory accepted by ``droid.user_interface.DataCollecter``."""

    def factory(episode_dir: str, droid_metadata: dict[str, Any]) -> DroidSidecarLifecycle:
        path = Path(episode_dir)
        return DroidSidecarLifecycle(
            path,
            options,
            metadata_builder(path, droid_metadata),
            block_closing=block_closing,
            abort_episode=abort_episode,
            safety_gate=safety_gate,
            calibration=calibration,
        )

    return factory
