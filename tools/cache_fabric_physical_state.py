#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.fabric_omni_bridge.cache import cache_episode
from fabric_droid.fabric_omni_bridge.encoder import FabricPhysicalEncoder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    encoder = FabricPhysicalEncoder(device=args.device)
    report = cache_episode(encoder, args.episode, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
