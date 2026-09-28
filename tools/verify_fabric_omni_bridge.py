#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.fabric_omni_bridge.encoder import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FABRIC_OMNI_ROOT,
    FabricPhysicalEncoder,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fabric-omni-root", type=Path, default=DEFAULT_FABRIC_OMNI_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--load-checkpoint", action="store_true")
    args = parser.parse_args()
    encoder = FabricPhysicalEncoder(args.fabric_omni_root, args.checkpoint)
    report = encoder.verify_contract()
    if args.load_checkpoint:
        report["checkpoint_load"] = encoder.load_checkpoint_contract()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
