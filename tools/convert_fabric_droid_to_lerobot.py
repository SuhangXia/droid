#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.conversion.lerobot import convert_session


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--minimum-duration-sec", type=float, default=1.0)
    args = parser.parse_args()
    splits = ("train", "validation") if args.include_validation else ("train",)
    report = convert_session(
        args.data_dir,
        args.output_root,
        args.repo_id,
        include_splits=splits,
        minimum_duration_sec=args.minimum_duration_sec,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
