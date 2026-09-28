#!/usr/bin/env python3
"""Migrate a Fabric-DROID dataset to the canonical target-tray task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.migration.task_labels import migrate_task_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and report the planned changes without writing files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = migrate_task_labels(args.dataset_root, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
