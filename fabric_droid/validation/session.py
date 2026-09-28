"""Session-level balance, integrity, and swatch leakage checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fabric_droid.validation.episode import validate_episode


def validate_session(session_dir: Path, *, minimum_duration_sec: float = 1.0) -> dict[str, Any]:
    episode_dirs = sorted({path.parent for path in session_dir.glob("**/trajectory.h5")})
    reports = [
        validate_episode(path, minimum_duration_sec=minimum_duration_sec, require_success=False) for path in episode_dirs
    ]
    tray_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    swatch_counts: Counter[str] = Counter()
    start_pose_counts: Counter[str] = Counter()
    swatch_splits: dict[str, set[str]] = defaultdict(set)
    for report in reports:
        metadata = report["metadata"]
        tray = str(metadata.get("destination_tray", "missing"))
        split = str(metadata.get("split", "missing"))
        swatch = str(metadata.get("swatch_uid", "missing"))
        tray_counts[tray] += 1
        split_counts[split] += 1
        swatch_counts[swatch] += 1
        start_pose_counts[str(metadata.get("start_pose_bucket", "missing"))] += 1
        swatch_splits[swatch].add(split)
    leaked = {swatch: sorted(splits) for swatch, splits in swatch_splits.items() if len(splits) > 1}
    successful = [report for report in reports if report["success"]]
    trainable = [
        report["episode_dir"]
        for report in reports
        if report["pass"] and report["success"] and report["metadata"].get("split") != "heldout_test"
    ]
    failures: list[str] = []
    if leaked:
        failures.append("swatch split leakage detected")
    if any(not report["pass"] for report in reports):
        failures.append("one or more episodes failed validation")
    return {
        "session_dir": str(session_dir.resolve()),
        "pass": not failures,
        "failures": failures,
        "episode_count": len(reports),
        "successful_episodes": len(successful),
        "failed_or_aborted_episodes": len(reports) - len(successful),
        "destination_tray_counts": dict(tray_counts),
        "split_counts": dict(split_counts),
        "swatch_counts": dict(swatch_counts),
        "start_pose_counts": dict(start_pose_counts),
        "swatch_split_leakage": leaked,
        "trainable_episodes": trainable,
        "episode_reports": reports,
    }
