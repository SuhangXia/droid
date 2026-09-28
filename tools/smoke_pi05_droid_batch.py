#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.conversion.pi05_smoke import run_pi05_batch_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--train-config-name", default="pi05_droid_finetune")
    args = parser.parse_args()
    report = run_pi05_batch_smoke(
        args.dataset_root,
        args.repo_id,
        checkpoint_dir=args.checkpoint_dir,
        train_config_name=args.train_config_name,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
