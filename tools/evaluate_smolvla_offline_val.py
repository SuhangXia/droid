#!/usr/bin/env python3
"""Compute deterministic held-out behaviour-cloning loss for SmolVLA checkpoints.

LeRobot v0.4.4 evaluates only Gym environments inside ``lerobot_train``.  It
does not provide an offline validation dataloader for real-robot datasets.  This
script evaluates a physical validation dataset after training and, crucially,
loads the checkpoint's saved preprocessor (including its *training* normalizer)
rather than recalculating one from the validation root.

SmolVLA's loss samples diffusion noise/time.  The random seed is reset before
each checkpoint so numbers are comparable checkpoint-to-checkpoint under the
same LeRobot/PyTorch build and validation data order.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


DEFAULT_RENAME_MAP = {
    "observation.images.exterior_1_left": "observation.images.camera1",
    "observation.images.wrist_left": "observation.images.camera2",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_model_dir(checkpoint: Path) -> Path:
    if (checkpoint / "pretrained_model" / "config.json").is_file():
        return checkpoint / "pretrained_model"
    if (checkpoint / "config.json").is_file():
        return checkpoint
    raise FileNotFoundError(
        f"{checkpoint}: expected a checkpoint directory containing pretrained_model/config.json "
        "or a pretrained_model directory"
    )


def _checkpoint_step(checkpoint: Path) -> int:
    try:
        return int(checkpoint.name)
    except ValueError:
        training_step = checkpoint / "training_state" / "training_step.json"
        if training_step.is_file():
            payload = json.loads(training_step.read_text(encoding="utf-8"))
            return int(payload["step"])
        return -1


def _iter_checkpoints(checkpoint: Path | None, checkpoints_dir: Path | None) -> list[Path]:
    if (checkpoint is None) == (checkpoints_dir is None):
        raise ValueError("provide exactly one of --checkpoint or --checkpoints-dir")
    if checkpoint is not None:
        return [checkpoint]
    assert checkpoints_dir is not None
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoints_dir}")
    candidates = [path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not candidates:
        raise ValueError(f"no numeric checkpoint directories found below {checkpoints_dir}")
    return sorted(candidates, key=_checkpoint_step)


def _make_dataloader(dataset: Any, policy: Any, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    from lerobot.datasets.sampler import EpisodeAwareSampler

    sampler = None
    if hasattr(policy.config, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=policy.config.drop_n_last_frames,
            shuffle=False,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False if sampler is None else False,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    dataset_root: Path,
    repo_id: str,
    rename_map: dict[str, str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    max_batches: int | None,
) -> dict[str, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    model_dir = _checkpoint_model_dir(checkpoint)
    _set_seed(seed)
    dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav")
    policy_cfg = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
    policy_cfg.pretrained_path = model_dir
    policy_cfg.device = str(device)

    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta, rename_map=rename_map)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_dir,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    dataloader = _make_dataloader(dataset, policy, batch_size, num_workers, device)

    total_loss = 0.0
    total_examples = 0
    batch_count = 0
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for batch in dataloader:
            processed = preprocessor(batch)
            loss, _ = policy.forward(processed)
            examples = int(next(value for value in batch.values() if torch.is_tensor(value)).shape[0])
            total_loss += float(loss.detach().float().item()) * examples
            total_examples += examples
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

    if total_examples == 0:
        raise RuntimeError(f"{checkpoint}: validation dataloader produced no batches")
    result = {
        "checkpoint": str(checkpoint),
        "model_dir": str(model_dir),
        "step": _checkpoint_step(checkpoint),
        "val_loss": total_loss / total_examples,
        "examples": total_examples,
        "batches": batch_count,
        "max_batches": max_batches,
        "seed": seed,
    }
    del policy, preprocessor, dataset, dataloader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint", type=Path)
    checkpoints.add_argument("--checkpoints-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Use only this many batches for a fast smoke test; zero means evaluate the full validation set.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing JSON report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_batches < 0:
        raise ValueError("batch size must be positive; worker/max-batch counts must be non-negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing report: {args.output}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"validation dataset root does not exist: {args.dataset_root}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_paths = _iter_checkpoints(args.checkpoint, args.checkpoints_dir)
    max_batches = None if args.max_batches == 0 else args.max_batches
    results = [
        evaluate_checkpoint(
            checkpoint,
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            rename_map=DEFAULT_RENAME_MAP,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
            max_batches=max_batches,
        )
        for checkpoint in checkpoint_paths
    ]
    payload = {
        "schema": "smolvla-offline-validation-v1",
        "dataset_root": str(args.dataset_root),
        "repo_id": args.repo_id,
        "results": results,
        "best_by_val_loss": min(results, key=lambda item: item["val_loss"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
