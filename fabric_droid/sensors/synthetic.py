"""Deterministic no-hardware streams used by safety-preserving smoke tests."""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from fabric_droid.sensors.ati import ATISample
from fabric_droid.sensors.base import SensorStream
from fabric_droid.sensors.camera import CameraFrame, CameraSpec


class SyntheticCameraStream(SensorStream):
    def __init__(self, spec: CameraSpec) -> None:
        super().__init__(spec.name)
        self.spec = spec
        self.frames: list[CameraFrame] = []
        self._lock = threading.Lock()
        self.dropped_frame_count = 0
        self.runtime_properties = {
            "width": spec.width,
            "height": spec.height,
            "fps": spec.fps,
            "exposure": spec.exposure,
            "white_balance": spec.white_balance,
            "synthetic": True,
        }

    def run(self) -> None:
        period_ns = int(1e9 / self.spec.fps)
        next_ns = time.monotonic_ns()
        while not self._stop_event.is_set():
            now = time.monotonic_ns()
            if now < next_ns:
                time.sleep(min((next_ns - now) / 1e9, 0.005))
                continue
            index = len(self.frames)
            x = np.linspace(0, 255, self.spec.width, dtype=np.uint8)
            y = np.linspace(0, 255, self.spec.height, dtype=np.uint8)[:, None]
            image = np.empty((self.spec.height, self.spec.width, 3), dtype=np.uint8)
            image[..., 0] = (x[None, :] + index * 3) % 255
            image[..., 1] = (y + index * 5) % 255
            image[..., 2] = (64 + index * 7) % 255
            timestamp = time.monotonic_ns()
            with self._lock:
                self.frames.append(CameraFrame(index, timestamp, timestamp, index * period_ns, image))
            next_ns += period_ns

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            count = len(self.frames)
            last = self.frames[-1].timestamp_monotonic_ns if self.frames else None
        return {
            "name": self.name,
            "spec": vars(self.spec),
            "runtime_properties": self.runtime_properties,
            "healthy": self.error is None and count > 0,
            "count": count,
            "last_timestamp_monotonic_ns": last,
            "dropped_frame_count": 0,
        }


class SyntheticATIStream(SensorStream):
    def __init__(self, sample_rate_hz: float = 500.0) -> None:
        super().__init__("ati_nano17")
        self.sample_rate_hz = sample_rate_hz
        self.samples: list[ATISample] = []
        self._lock = threading.Lock()

    def run(self) -> None:
        period_ns = int(1e9 / self.sample_rate_hz)
        next_ns = time.monotonic_ns()
        while not self._stop_event.is_set():
            now = time.monotonic_ns()
            if now < next_ns:
                time.sleep(min((next_ns - now) / 1e9, 0.001))
                continue
            index = len(self.samples)
            phase = index / self.sample_rate_hz
            wrench = (
                0.1 * math.sin(phase * 3),
                0.1 * math.cos(phase * 2),
                1.0 + 0.5 * math.sin(phase * 5),
                0.01 * math.sin(phase),
                0.01 * math.cos(phase),
                0.005 * math.sin(phase * 4),
            )
            timestamp = time.monotonic_ns()
            with self._lock:
                self.samples.append(ATISample(timestamp, time.time_ns(), index, *wrench, "synthetic", index * period_ns))
            next_ns += period_ns

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            count = len(self.samples)
            last = self.samples[-1].timestamp_monotonic_ns if self.samples else None
        return {
            "name": self.name,
            "healthy": self.error is None and count > 0,
            "count": count,
            "last_timestamp_monotonic_ns": last,
            "sample_rate_hz": self.sample_rate_hz,
            "conflate": False,
            "synthetic": True,
        }
