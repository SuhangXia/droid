from __future__ import annotations

import argparse
import json
from pathlib import Path

from .services.lerobot_smoke import smoke_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifest-based DROID to LeRobot smoke validation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--max-segments", type=int, default=2)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if not args.smoke_only:
        raise SystemExit(
            "Only --smoke-only is implemented; formal dataset conversion/training is intentionally disabled."
        )
    result = smoke_manifest(
        args.manifest.resolve(),
        data_root=args.data_root.resolve() if args.data_root else None,
        max_segments=args.max_segments,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
