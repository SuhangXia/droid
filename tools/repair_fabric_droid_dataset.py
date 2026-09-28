#!/usr/bin/env python3
"""Repair D435 MP4 timelines and label Fabric-DROID episodes non-destructively."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.repair import repair_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    args = parser.parse_args()
    manifest = repair_dataset(args.source_root, args.destination_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
