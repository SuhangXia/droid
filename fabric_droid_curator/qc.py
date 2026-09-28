from __future__ import annotations

import argparse
import json

from sqlalchemy import select

from .config import load_config
from .db.models import Episode, QCResult
from .db.session import Database
from .services.indexer import EpisodeIndexer


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute Fabric-DROID signal QC and event proposals.")
    parser.add_argument("--data-root")
    parser.add_argument("--config", default="configs/curator.yaml")
    parser.add_argument("--episode", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, data_root=args.data_root)
    database = Database(config.database_url)
    summary = EpisodeIndexer(config, database).scan(
        episode_ids=set(args.episode) or None,
        generate_proxies=False,
        force=True,
        progress=print,
    )
    with database.session() as db:
        episodes = {item.id: item.episode_id for item in db.scalars(select(Episode))}
        results = [
            {
                "episode_id": episodes.get(item.episode_pk),
                "severity": item.severity,
                "scores": json.loads(item.scores_json),
                "hard_failures": json.loads(item.hard_failures_json),
                "warnings": json.loads(item.warnings_json),
            }
            for item in db.scalars(select(QCResult))
        ]
    print(json.dumps({"scan": summary, "qc": results}, indent=2, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
