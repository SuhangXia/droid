"""Construct and validate a real OpenPI DroidInputs batch without training."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def run_pi05_batch_smoke(
    dataset_root: Path,
    repo_id: str,
    frame_index: int = 0,
    *,
    checkpoint_dir: Path | None = None,
    train_config_name: str = "pi05_droid_finetune",
) -> dict[str, Any]:
    cache_root = Path(
        os.environ.get("FABRIC_DROID_HF_CACHE", str(Path(tempfile.gettempdir()) / "fabric_droid_hf_cache"))
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    try:
        import datasets

        datasets.config.HF_DATASETS_CACHE = str(cache_root / "datasets")
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from openpi.models import model as model_api
        from openpi.policies.droid_policy import DroidInputs
    except ImportError as exc:
        raise RuntimeError("run this smoke test inside the existing OpenPI environment") from exc
    dataset = LeRobotDataset(repo_id, root=dataset_root)
    if len(dataset) == 0:
        raise ValueError("LeRobot dataset is empty")
    first = dataset[frame_index]
    last = dataset[len(dataset) - 1]

    def image_array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    action = image_array(first["actions"]).astype(np.float32)
    horizon_actions = np.repeat(action[None, :], 16, axis=0)
    raw = {
        "observation/exterior_image_1_left": image_array(first["exterior_image_1_left"]),
        "observation/wrist_image_left": image_array(first["wrist_image_left"]),
        "observation/joint_position": image_array(first["joint_position"]),
        "observation/gripper_position": image_array(first["gripper_position"]),
        "actions": horizon_actions,
        "prompt": first["task"],
    }
    transformed = DroidInputs(model_type=model_api.ModelType.PI05)(raw)
    expected_images = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    failures: list[str] = []
    if transformed["state"].shape != (8,):
        failures.append(f"state shape is {transformed['state'].shape}, expected (8,)")
    if transformed["actions"].shape != (16, 8):
        failures.append(f"actions shape is {transformed['actions'].shape}, expected (16, 8)")
    if set(transformed["image"]) != expected_images:
        failures.append(f"image keys are {sorted(transformed['image'])}")
    masks = {key: bool(value) for key, value in transformed["image_mask"].items()}
    if masks != {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}:
        failures.append(f"unexpected image masks: {masks}")
    if not np.isfinite(transformed["state"]).all() or not np.isfinite(transformed["actions"]).all():
        failures.append("state/actions contain NaN or Inf")
    report = {
        "pass": not failures,
        "failures": failures,
        "dataset_length": len(dataset),
        "first_task": str(first["task"]),
        "last_task": str(last["task"]),
        "first_frame_image_shapes": {
            "exterior_image_1_left": list(image_array(first["exterior_image_1_left"]).shape),
            "wrist_image_left": list(image_array(first["wrist_image_left"]).shape),
        },
        "last_frame_action_shape": list(image_array(last["actions"]).shape),
        "pi05": {
            "state_shape": list(transformed["state"].shape),
            "actions_shape": list(transformed["actions"].shape),
            "image_shapes": {key: list(value.shape) for key, value in transformed["image"].items()},
            "image_masks": masks,
            "prompt": transformed["prompt"],
            "configured_action_dim": 32,
            "configured_action_horizon": 16,
        },
        "forward_executed": False,
        "forward_reason": (
            "This command validates the data transform only; model forward requires GPU and pi05_droid assets."
        ),
    }
    if checkpoint_dir is not None:
        if not checkpoint_dir.is_dir():
            report["failures"].append(f"checkpoint directory does not exist: {checkpoint_dir}")
        else:
            import jax

            gpu_available = any(device.platform == "gpu" for device in jax.devices())
            assets_available = (checkpoint_dir / "params").exists() and (checkpoint_dir / "assets").exists()
            if not gpu_available:
                report["forward_reason"] = "JAX GPU device is unavailable; forward was not attempted."
            elif not assets_available:
                report["forward_reason"] = "checkpoint params or normalization assets are unavailable."
            else:
                from openpi.policies import policy_config
                from openpi.training import config as training_config

                policy = policy_config.create_trained_policy(
                    training_config.get_config(train_config_name),
                    checkpoint_dir,
                )
                policy_input = {key: value for key, value in raw.items() if key != "actions"}
                output = policy.infer(policy_input)
                output_actions = np.asarray(output["actions"])
                report["forward_executed"] = True
                report["forward_reason"] = ""
                report["forward_action_shape"] = list(output_actions.shape)
                if output_actions.shape[-1] != 8:
                    report["failures"].append(f"forward returned action width {output_actions.shape[-1]}, expected 8")
        report["pass"] = not report["failures"]
    return report
