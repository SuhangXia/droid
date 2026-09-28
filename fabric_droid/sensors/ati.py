"""ATI Nano17 ZMQ acquisition without downsampling."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fabric_droid.sensors.base import SensorStream

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
ATI_ZERO_EPSILON = 1e-9


@dataclass(frozen=True)
class ATISample:
    timestamp_monotonic_ns: int
    timestamp_wall_ns: int
    sample_index: int
    fx: float
    fy: float
    fz: float
    tx: float
    ty: float
    tz: float
    device_status: str = "ok"
    device_timestamp_ns: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_ati_payload(payload: bytes | str) -> tuple[np.ndarray, int, str]:
    """Parse JSON or ASCII ATI payloads used by the Clothumi/Octopi publishers."""

    text = payload.decode("utf-8", errors="strict") if isinstance(payload, bytes) else payload
    text = text.strip()
    device_timestamp_ns = -1
    status = "ok"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("wrench", "force_torque", "ft"):
            if key in parsed:
                parsed = parsed[key]
                break
        if isinstance(parsed, dict):
            lower = {str(key).lower(): value for key, value in parsed.items()}
            values = [lower[key] for key in ("fx", "fy", "fz", "tx", "ty", "tz")]
            device_timestamp_ns = int(lower.get("device_timestamp_ns", -1))
            status = str(lower.get("status", "ok"))
            return np.asarray(values, dtype=np.float64), device_timestamp_ns, status
        if isinstance(parsed, list):
            values = parsed
        else:
            values = []
    elif isinstance(parsed, list):
        values = parsed
    else:
        values = [float(match) for match in _NUMBER.findall(text)]
    if len(values) < 6:
        raise ValueError(f"ATI payload contains {len(values)} numeric channels; expected at least 6")
    return np.asarray(values[-6:], dtype=np.float64), device_timestamp_ns, status


class ATIStream(SensorStream):
    """One ZMQ subscriber sample becomes one persisted row; CONFLATE is forbidden."""

    def __init__(
        self,
        endpoint: str = "tcp://192.168.1.20:5555",
        topic: str = "",
        recv_timeout_ms: int = 1000,
        *,
        max_buffer_samples: int | None = None,
    ) -> None:
        super().__init__("ati_nano17")
        if max_buffer_samples is not None and max_buffer_samples < 1:
            raise ValueError("max_buffer_samples must be at least 1")
        self.endpoint = endpoint
        self.topic = topic
        self.recv_timeout_ms = recv_timeout_ms
        self.max_buffer_samples = max_buffer_samples
        self.samples: list[ATISample] = []
        self._total_sample_count = 0
        self._lock = threading.Lock()
        self._gap_monitor_start_ns: int | None = None
        self._max_inter_sample_gap_ns = 0

    def run(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq is required for ATI ZMQ capture") from exc
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        try:
            socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
            socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
            socket.setsockopt(zmq.CONFLATE, 0)
            socket.connect(self.endpoint)
            while not self._stop_event.is_set():
                try:
                    parts = socket.recv_multipart()
                except zmq.Again:
                    continue
                arrival_ns = time.monotonic_ns()
                wall_ns = time.time_ns()
                values, device_ns, status = parse_ati_payload(parts[-1])
                self._append(
                    values,
                    arrival_ns=arrival_ns,
                    wall_ns=wall_ns,
                    device_timestamp_ns=device_ns,
                    status=status,
                )
        finally:
            socket.close(linger=0)
            context.term()

    def _append(
        self,
        values: np.ndarray,
        *,
        arrival_ns: int | None = None,
        wall_ns: int | None = None,
        device_timestamp_ns: int = -1,
        status: str = "ok",
    ) -> None:
        """Append one parsed packet and update the episode-local gap latch."""

        wrench = np.asarray(values, dtype=np.float64)
        if wrench.shape != (6,):
            raise ValueError(f"ATI wrench must have shape (6,), got {wrench.shape}")
        arrival_ns = time.monotonic_ns() if arrival_ns is None else int(arrival_ns)
        wall_ns = time.time_ns() if wall_ns is None else int(wall_ns)
        with self._lock:
            if self._gap_monitor_start_ns is not None:
                previous_ns = (
                    self.samples[-1].timestamp_monotonic_ns
                    if self.samples
                    and self.samples[-1].timestamp_monotonic_ns
                    >= self._gap_monitor_start_ns
                    else self._gap_monitor_start_ns
                )
                self._max_inter_sample_gap_ns = max(
                    self._max_inter_sample_gap_ns,
                    arrival_ns - previous_ns,
                )
            self.samples.append(
                ATISample(
                    arrival_ns,
                    wall_ns,
                    self._total_sample_count,
                    *wrench.tolist(),
                    str(status),
                    int(device_timestamp_ns),
                )
            )
            self._total_sample_count += 1
            if (
                self.max_buffer_samples is not None
                and len(self.samples) > self.max_buffer_samples
            ):
                del self.samples[
                    : len(self.samples) - self.max_buffer_samples
                ]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            count = self._total_sample_count
            buffered_count = len(self.samples)
            last_ns = self.samples[-1].timestamp_monotonic_ns if self.samples else None
            max_gap_ns = self._max_inter_sample_gap_ns
        return {
            "name": self.name,
            "healthy": self.error is None and count > 0,
            "count": count,
            "buffered_count": buffered_count,
            "max_buffer_samples": self.max_buffer_samples,
            "last_timestamp_monotonic_ns": last_ns,
            "episode_max_inter_sample_gap_sec": max_gap_ns / 1e9,
            "endpoint": self.endpoint,
            "conflate": False,
        }

    def latest_samples(self, limit: int = 2500) -> list[ATISample]:
        if limit < 1:
            return []
        with self._lock:
            return list(self.samples[-limit:])

    def reset_gap_monitor(self, start_monotonic_ns: int) -> None:
        """Start a new episode-local historical packet-gap monitor."""

        if start_monotonic_ns <= 0:
            raise ValueError("gap monitor start must be a positive timestamp")
        with self._lock:
            self._gap_monitor_start_ns = start_monotonic_ns
            latest_ns = (
                self.samples[-1].timestamp_monotonic_ns
                if self.samples
                else None
            )
            self._max_inter_sample_gap_ns = (
                latest_ns - start_monotonic_ns
                if latest_ns is not None
                and latest_ns >= start_monotonic_ns
                else 0
            )

    def samples_between(
        self,
        start_monotonic_ns: int,
        end_monotonic_ns: int,
    ) -> list[ATISample]:
        """Return samples within the episode time boundary, without downsampling."""

        if end_monotonic_ns < start_monotonic_ns:
            raise ValueError("ATI sample interval end precedes start")
        with self._lock:
            return [
                sample
                for sample in self.samples
                if start_monotonic_ns
                <= sample.timestamp_monotonic_ns
                <= end_monotonic_ns
            ]

    def readiness(
        self,
        *,
        minimum_samples: int = 16,
        zero_window_samples: int = 64,
        max_age_sec: float = 1.0,
        max_gap_sec: float | None = None,
        since_monotonic_ns: int | None = None,
    ) -> dict[str, Any]:
        """Report whether ATI is delivering fresh, finite, non-placeholder data."""

        if minimum_samples < 1 or zero_window_samples < 1:
            raise ValueError("ATI readiness sample windows must be positive")
        if max_age_sec <= 0:
            raise ValueError("ATI max_age_sec must be positive")
        if max_gap_sec is not None and max_gap_sec <= 0:
            raise ValueError("ATI max_gap_sec must be positive")
        window_size = max(minimum_samples, zero_window_samples)
        with self._lock:
            count = self._total_sample_count
            samples = list(self.samples[-window_size:])
            max_gap_ns = self._max_inter_sample_gap_ns
        if since_monotonic_ns is not None:
            samples = [
                sample
                for sample in samples
                if sample.timestamp_monotonic_ns >= since_monotonic_ns
            ]
        samples = samples[-window_size:]
        gate_count = count if since_monotonic_ns is None else len(samples)
        latest_ns = samples[-1].timestamp_monotonic_ns if samples else None
        age_sec = (
            None
            if latest_ns is None
            else max(0.0, (time.monotonic_ns() - latest_ns) / 1e9)
        )
        values = np.asarray(
            [
                [sample.fx, sample.fy, sample.fz, sample.tx, sample.ty, sample.tz]
                for sample in samples[-zero_window_samples:]
            ],
            dtype=np.float64,
        ).reshape(-1, 6)
        finite = bool(values.size and np.isfinite(values).all())
        nonzero = bool(
            finite and np.any(np.abs(values) > ATI_ZERO_EPSILON)
        )
        statuses_ok = bool(
            samples
            and all(sample.device_status.lower() == "ok" for sample in samples[-minimum_samples:])
        )
        reasons: list[str] = []
        if self.error is not None:
            reasons.append(f"stream error: {self.error}")
        if gate_count < minimum_samples:
            qualifier = (
                ""
                if since_monotonic_ns is None
                else " since episode start"
            )
            reasons.append(
                f"only {gate_count}/{minimum_samples} samples received{qualifier}"
            )
        if age_sec is None or age_sec > max_age_sec:
            reasons.append(
                "no samples received"
                if age_sec is None
                else f"last sample is stale ({age_sec:.3f}s)"
            )
        if values.size and not finite:
            reasons.append("wrench contains NaN/Inf")
        if values.size and finite and not nonzero:
            reasons.append("recent wrench samples are all zero")
        if samples and not statuses_ok:
            reasons.append("ATI device status is not ok")
        max_gap_sec_observed = max_gap_ns / 1e9
        if (
            max_gap_sec is not None
            and max_gap_sec_observed > max_gap_sec
        ):
            reasons.append(
                "historical packet gap "
                f"{max_gap_sec_observed:.3f}s exceeds {max_gap_sec:.3f}s"
            )
        return {
            "ready": not reasons,
            "reasons": reasons,
            "count": count,
            "gate_count": gate_count,
            "latest_timestamp_monotonic_ns": latest_ns,
            "age_sec": age_sec,
            "finite": finite,
            "nonzero": nonzero,
            "statuses_ok": statuses_ok,
            "minimum_samples": minimum_samples,
            "zero_window_samples": zero_window_samples,
            "max_age_sec": max_age_sec,
            "max_gap_sec": max_gap_sec,
            "max_gap_sec_observed": max_gap_sec_observed,
            "since_monotonic_ns": since_monotonic_ns,
        }
