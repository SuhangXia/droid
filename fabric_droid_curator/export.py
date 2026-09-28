from __future__ import annotations

import argparse
import json

from .config import load_config
from .db.session import Database
from .services.exports import ManifestExporter


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an immutable Fabric-DROID segment manifest.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--config", default="configs/curator.yaml")
    parser.add_argument("--include-heldout", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = ManifestExporter(config, Database(config.database_url)).export(
        args.version,
        include_heldout=args.include_heldout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
