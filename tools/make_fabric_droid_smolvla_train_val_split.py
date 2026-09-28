#!/usr/bin/env python3
"""Create a leakage-free, source-episode-level Fabric-DROID train/val split.

The original v001 curator export intentionally contains only training rows.  A
single DROID teleoperation episode is represented by two derived segments
(``grasp_probe`` plus either ``leave_on_rack`` or ``remove_to_basket``), so
splitting its segments independently would leak nearly identical observations
into validation.  This tool assigns whole *source episodes* to one split.

It writes immutable train/val parquet manifests plus ``split.json``.  If the
existing full LeRobot conversion report is provided, ``split.json`` additionally
records the corresponding episode indices and frame totals as an audit trail.

Example:
    python tools/make_fabric_droid_smolvla_train_val_split.py \
      --manifest curation/exports/fabric_droid_v001/manifest.parquet \
      --conversion-report /home/suhang/datasets2/smolvla/fabric_droid_v001_v3/smolvla_conversion_report.json \
      --output-dir curation/exports/fabric_droid_v001_smolvla_trainval_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "segment_id",
    "source_episode_id",
    "segment_type",
    "action_branch",
    "prompt",
    "split",
    "dataset_decision",
}
SOURCE_DAY_PATTERN = re.compile(r"^episode_(\d{8})_\d{6}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: str, source_episode_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{source_episode_id}".encode("utf-8")).hexdigest()


def source_day(source_episode_id: str) -> str:
    match = SOURCE_DAY_PATTERN.fullmatch(source_episode_id)
    if match is None:
        raise ValueError(
            f"unexpected source_episode_id {source_episode_id!r}; expected episode_YYYYMMDD_HHMMSS"
        )
    return match.group(1)


def _allocate_quotas(strata: dict[tuple[str, str], list[str]], val_fraction: float) -> dict[tuple[str, str], int]:
    """Allocate an exact global val count while preserving collection-day/task mix."""
    total = sum(len(items) for items in strata.values())
    target = int(round(total * val_fraction))
    if not 1 <= target < total:
        raise ValueError(
            f"val fraction {val_fraction} yields {target} held-out source episodes for {total}; "
            "choose a fraction that leaves at least one source in each split"
        )

    ideals = {key: len(items) * val_fraction for key, items in strata.items()}
    quotas = {key: min(len(strata[key]) - 1, math.floor(ideal)) for key, ideal in ideals.items()}
    remaining = target - sum(quotas.values())
    ordered = sorted(
        strata,
        key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key[0], key[1]),
    )
    for key in ordered:
        if remaining == 0:
            break
        if quotas[key] < len(strata[key]) - 1:
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("unable to allocate the requested validation quota")
    return quotas


def load_conversion_report(path: Path, expected_segment_ids: set[str]) -> tuple[dict[str, int], dict[str, int]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{path}: expected a conversion report with an episodes list")

    index_by_segment: dict[str, int] = {}
    frames_by_segment: dict[str, int] = {}
    for episode in episodes:
        segment_id = str(episode["segment_id"])
        if segment_id in index_by_segment:
            raise ValueError(f"{path}: duplicate segment_id {segment_id}")
        index_by_segment[segment_id] = int(episode["output_episode_index"])
        frames_by_segment[segment_id] = int(episode["frame_count"])

    if set(index_by_segment) != expected_segment_ids:
        missing = sorted(expected_segment_ids - set(index_by_segment))
        extra = sorted(set(index_by_segment) - expected_segment_ids)
        raise ValueError(
            f"{path}: conversion report does not match manifest; missing={missing[:3]}, extra={extra[:3]}"
        )
    return index_by_segment, frames_by_segment


def build_split(
    manifest: Path,
    output_dir: Path,
    *,
    val_fraction: float,
    seed: str,
    conversion_report: Path | None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing split output: {output_dir}")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between zero and one")

    table = pq.read_table(manifest)
    missing = REQUIRED_COLUMNS - set(table.column_names)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    rows = table.to_pylist()
    if not rows:
        raise ValueError("manifest is empty")

    invalid = [
        str(row["segment_id"])
        for row in rows
        if str(row["split"]) != "train" or str(row["dataset_decision"]) != "keep"
    ]
    if invalid:
        raise ValueError(
            "this splitter expects the immutable v001 train-only manifest; bad segment IDs: "
            + ", ".join(invalid[:5])
        )

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_episode_id"])].append(row)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source_episode_id, source_rows in by_source.items():
        branches = {str(row["action_branch"]) for row in source_rows}
        if len(branches) != 1:
            raise ValueError(
                f"{source_episode_id}: all derived segments must have one action branch, got {sorted(branches)}"
            )
        branch = next(iter(branches))
        strata[(source_day(source_episode_id), branch)].append(source_episode_id)

    quotas = _allocate_quotas(strata, val_fraction)
    val_sources: set[str] = set()
    stratum_report: list[dict[str, Any]] = []
    for key in sorted(strata):
        source_ids = sorted(strata[key], key=lambda item: stable_rank(seed, item))
        selected = source_ids[: quotas[key]]
        val_sources.update(selected)
        stratum_report.append(
            {
                "collection_day": key[0],
                "action_branch": key[1],
                "source_episode_count": len(source_ids),
                "val_source_episode_count": len(selected),
                "val_source_episode_ids": sorted(selected),
            }
        )

    train_sources = set(by_source) - val_sources
    if not train_sources or not val_sources or train_sources & val_sources:
        raise RuntimeError("invalid source-level split assignment")

    assignments = {source_id: ("val" if source_id in val_sources else "train") for source_id in by_source}
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    segment_assignments: dict[str, str] = {}
    for row in rows:
        row_copy = dict(row)
        split = assignments[str(row["source_episode_id"])]
        row_copy["split"] = split
        segment_id = str(row["segment_id"])
        segment_assignments[segment_id] = split
        (train_rows if split == "train" else val_rows).append(row_copy)

    if len(segment_assignments) != len(rows):
        raise RuntimeError("manifest contains duplicate segment IDs")

    expected_segment_ids = set(segment_assignments)
    indices: dict[str, list[int]] = {}
    frame_totals: dict[str, int] = {}
    if conversion_report is not None:
        index_by_segment, frames_by_segment = load_conversion_report(conversion_report, expected_segment_ids)
        for split in ("train", "val"):
            segment_ids = [segment_id for segment_id, assigned in segment_assignments.items() if assigned == split]
            indices[split] = sorted(index_by_segment[segment_id] for segment_id in segment_ids)
            frame_totals[split] = sum(frames_by_segment[segment_id] for segment_id in segment_ids)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{uuid4().hex}"
    staging.mkdir()
    try:
        pq.write_table(pa.Table.from_pylist(train_rows, schema=table.schema), staging / "train_manifest.parquet")
        pq.write_table(pa.Table.from_pylist(val_rows, schema=table.schema), staging / "val_manifest.parquet")

        report: dict[str, Any] = {
            "schema": "fabric-droid-smolvla-source-episode-split-v1",
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256(manifest),
            "seed": seed,
            "val_fraction": val_fraction,
            "source_episode_assignments": dict(sorted(assignments.items())),
            "segment_assignments": dict(sorted(segment_assignments.items())),
            "counts": {
                "source_episodes": {"total": len(by_source), "train": len(train_sources), "val": len(val_sources)},
                "segments": {"total": len(rows), "train": len(train_rows), "val": len(val_rows)},
                "action_branch_sources": {
                    "train": dict(sorted(Counter(next(iter({str(row['action_branch']) for row in by_source[source_id]})) for source_id in train_sources).items())),
                    "val": dict(sorted(Counter(next(iter({str(row['action_branch']) for row in by_source[source_id]})) for source_id in val_sources).items())),
                },
            },
            "strata": stratum_report,
            "artifacts": {"train_manifest": "train_manifest.parquet", "val_manifest": "val_manifest.parquet"},
            "validation": {
                "source_episode_overlap": sorted(train_sources & val_sources),
                "all_segments_assigned_once": len(segment_assignments) == len(rows),
            },
        }
        if conversion_report is not None:
            report["conversion_report"] = str(conversion_report.resolve())
            report["dataset_episode_indices"] = indices
            report["frame_totals"] = frame_totals
            report["validation"]["episode_index_overlap"] = sorted(set(indices["train"]) & set(indices["val"]))

        (staging / "split.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conversion-report", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", default="fabric-droid-smolvla-trainval-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_split(
        args.manifest,
        args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        conversion_report=args.conversion_report,
    )
    print(json.dumps(report["counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
