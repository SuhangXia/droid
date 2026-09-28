#!/usr/bin/env python3
"""Receive and summarize an ATI ZMQ stream without commanding any robot."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.sensors.ati import ATIStream
from fabric_droid.sync.clock import analyze_timestamps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://192.168.1.20:5555")
    parser.add_argument("--duration-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stream = ATIStream(args.endpoint, recv_timeout_ms=250)
    stream.reset_gap_monitor(time.monotonic_ns())
    stream.start()
    deadline = time.monotonic() + args.duration_sec
    try:
        while time.monotonic() < deadline and stream.error is None:
            time.sleep(0.02)
    finally:
        try:
            stream.stop(timeout=2.0)
        except RuntimeError:
            pass
    samples = list(stream.samples)
    clock = analyze_timestamps([sample.timestamp_monotonic_ns for sample in samples])
    readiness = stream.readiness(max_gap_sec=1.0)
    report = {
        "pass": bool(readiness["ready"]),
        "endpoint": args.endpoint,
        "error": None if stream.error is None else f"{type(stream.error).__name__}: {stream.error}",
        "readiness": readiness,
        "clock": clock.to_dict(),
        "first": samples[0].to_dict() if samples else None,
        "last": samples[-1].to_dict() if samples else None,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
