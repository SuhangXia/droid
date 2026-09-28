"""Host-monotonic timebase and diagnostics."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


class HostClock:
    """The only clock used to align Fabric-DROID streams."""

    @staticmethod
    def monotonic_ns() -> int:
        return time.monotonic_ns()

    @staticmethod
    def wall_ns() -> int:
        return time.time_ns()


@dataclass(frozen=True)
class StreamClockReport:
    count: int
    start_ns: int | None
    end_ns: int | None
    duration_sec: float
    measured_hz: float
    monotonic: bool
    duplicate_count: int
    max_gap_ms: float

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


def analyze_timestamps(values: Sequence[int] | np.ndarray) -> StreamClockReport:
    timestamps = np.asarray(values, dtype=np.int64)
    if timestamps.size == 0:
        return StreamClockReport(0, None, None, 0.0, 0.0, True, 0, 0.0)
    differences = np.diff(timestamps)
    duration = max(0.0, float(timestamps[-1] - timestamps[0]) / 1e9)
    hz = float(timestamps.size - 1) / duration if timestamps.size > 1 and duration > 0 else 0.0
    return StreamClockReport(
        count=int(timestamps.size),
        start_ns=int(timestamps[0]),
        end_ns=int(timestamps[-1]),
        duration_sec=duration,
        measured_hz=hz,
        monotonic=bool(np.all(differences > 0)),
        duplicate_count=int(np.count_nonzero(differences == 0)),
        max_gap_ms=float(differences.max() / 1e6) if differences.size else 0.0,
    )
