#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.validation.episode import validate_episode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--minimum-duration-sec", type=float, default=1.0)
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()
    report = validate_episode(
        args.episode,
        minimum_duration_sec=args.minimum_duration_sec,
        require_success=not args.allow_failure,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
