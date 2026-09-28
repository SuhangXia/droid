from __future__ import annotations

import argparse
import json

from .config import load_config
from .db.session import Database
from .services.indexer import EpisodeIndexer


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally index read-only Fabric-DROID episodes.")
    parser.add_argument("--data-root")
    parser.add_argument("--config", default="configs/curator.yaml")
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--skip-proxies", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, data_root=args.data_root)
    database = Database(config.database_url)
    summary = EpisodeIndexer(config, database).scan(
        episode_ids=set(args.episode) or None,
        generate_proxies=not args.skip_proxies,
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
