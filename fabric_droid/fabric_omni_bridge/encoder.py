"""Read-only adapter around Fabric-Omni's frozen Final-170 TeacherBV2."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fabric_droid.io_utils import sha256_file
from fabric_droid.schemas import ForceCalibration

DEFAULT_FABRIC_OMNI_ROOT = Path("/home/suhang/projects/fabric-omni")
DEFAULT_CHECKPOINT = DEFAULT_FABRIC_OMNI_ROOT / (
    "artifacts/pilots/v2_physical_final170_20260715_071119/exports/final170_physical_teacher/model_only.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "132aeab7a9f2e736ed5e8c83c86e3280ce413590c2b1738534c6d1a5dd0f7f0e"


@dataclass(frozen=True)
class PhysicalState:
    global_tactile_latent: np.ndarray
    predicted_normal_force: float
    contact_probability: float
    contact_phase: int
    contact_area_proxy: float
    softness_score: float
    uncertainty: float
    tactile_valid: bool
    force_valid: bool
    physical_state_valid: bool
    checkpoint_sha256: str
    config_sha256: str
    semantics: dict[str, str]

    def serializable_metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value["global_tactile_latent"] = {
            "shape": list(self.global_tactile_latent.shape),
            "dtype": str(self.global_tactile_latent.dtype),
        }
        return value


@contextlib.contextmanager
def _read_only_import(root: Path):
    """Import from Fabric-Omni without allowing Python bytecode in that tree."""

    source = str(root / "src")
    old_dont_write = sys.dont_write_bytecode
    old_env = os.environ.get("PYTHONDONTWRITEBYTECODE")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, source)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == source:
            sys.path.pop(0)
        sys.dont_write_bytecode = old_dont_write
        if old_env is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = old_env


class FabricPhysicalEncoder:
    """Frozen one-window inference facade.

    `softness_score` is the model's flattening/compression response proxy. It
    must not be interpreted as a measured or classified material softness.
    """

    def __init__(
        self,
        fabric_omni_root: Path = DEFAULT_FABRIC_OMNI_ROOT,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        *,
        device: str = "cpu",
        verify_hash: bool = True,
    ) -> None:
        self.fabric_omni_root = fabric_omni_root.resolve()
        # Preserve the lexical path: the official export is a symlink whose
        # target lives on the datasets volume.
        self.checkpoint_path = checkpoint_path.absolute()
        self.checkpoint_target = checkpoint_path.resolve()
        self.device = device
        self._model = None
        self._batch_type = None
        self._checkpoint_hash: str | None = None
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        if self.fabric_omni_root not in self.checkpoint_path.parents:
            raise ValueError("checkpoint must be a read-only input under the selected Fabric-Omni root")
        if verify_hash:
            digest = sha256_file(self.checkpoint_path)
            if digest != EXPECTED_CHECKPOINT_SHA256:
                raise ValueError(f"unexpected Final-170 checkpoint hash: {digest}")
            self._checkpoint_hash = digest

    @property
    def checkpoint_sha256(self) -> str:
        if self._checkpoint_hash is None:
            self._checkpoint_hash = sha256_file(self.checkpoint_path)
        return self._checkpoint_hash

    def verify_contract(self) -> dict[str, Any]:
        export_dir = self.checkpoint_path.parent
        preprocessing = json.loads((export_dir / "preprocessing_contract.json").read_text(encoding="utf-8"))
        tokens = json.loads((export_dir / "token_contract.json").read_text(encoding="utf-8"))
        with _read_only_import(self.fabric_omni_root):
            from fabric_omni.models.full_model.teacher_v2 import TeacherBV2
            from fabric_omni.schemas.observation_v2 import SingleObservationBatch

        return {
            "pass": preprocessing.get("frames") == 16
            and preprocessing.get("single_press") is True
            and tokens.get("observed_surface_only") is True,
            "fabric_omni_root": str(self.fabric_omni_root),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_target": str(self.checkpoint_target),
            "checkpoint_sha256": self.checkpoint_sha256,
            "teacher_class": f"{TeacherBV2.__module__}.{TeacherBV2.__name__}",
            "observation_class": f"{SingleObservationBatch.__module__}.{SingleObservationBatch.__name__}",
            "preprocessing_contract": preprocessing,
            "token_contract": tokens,
            "writes_to_fabric_omni": False,
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for Fabric-Omni inference") from exc
        with _read_only_import(self.fabric_omni_root):
            from fabric_omni.models.full_model.teacher_v2 import TeacherBV2
            from fabric_omni.schemas.observation_v2 import SingleObservationBatch

            model = TeacherBV2()
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = {
            "fusion.rgb_appearance_common_proj.bias",
            "fusion.rgb_appearance_common_proj.weight",
            "fusion.rgb_texture_common_proj.bias",
            "fusion.rgb_texture_common_proj.weight",
        }
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint contract mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        model.eval()
        for parameter in model.parameters():
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter):
                parameter.requires_grad_(False)
        model.to(self.device)
        self._model = model
        self._batch_type = SingleObservationBatch

    def load_checkpoint_contract(self) -> dict[str, Any]:
        """Instantiate the frozen teacher and validate every checkpoint key."""

        self._load()
        assert self._model is not None
        import torch

        parameters = [
            parameter
            for parameter in self._model.parameters()
            if not isinstance(parameter, torch.nn.parameter.UninitializedParameter)
        ]
        return {
            "loaded": True,
            "device": self.device,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "requires_grad_count": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            "training": self._model.training,
            "checkpoint_sha256": self.checkpoint_sha256,
            "allowed_missing_compatibility_keys": 4,
            "unexpected_keys": 0,
        }

    @staticmethod
    def _frames(gelsight_window: np.ndarray, size: int = 224, count: int = 16) -> np.ndarray:
        frames = np.asarray(gelsight_window)
        if frames.ndim != 4:
            raise ValueError("gelsight_window must be [T,H,W,3] or [T,3,H,W]")
        if frames.shape[-1] == 3:
            frames = np.moveaxis(frames, -1, 1)
        if frames.shape[1] != 3 or frames.shape[0] < 2:
            raise ValueError("gelsight_window must contain at least two RGB frames")
        indices = np.linspace(0, frames.shape[0] - 1, count).round().astype(int)
        selected = frames[indices].astype(np.float32)
        if selected.max() > 1.0:
            selected /= 255.0
        import torch

        tensor = torch.from_numpy(selected)
        if tuple(tensor.shape[-2:]) != (size, size):
            tensor = torch.nn.functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        return tensor.numpy()

    @staticmethod
    def _normal_force(
        ati_window: np.ndarray | None,
        calibration: ForceCalibration | None,
    ) -> tuple[np.ndarray, bool]:
        if ati_window is None or calibration is None or not calibration.normal_force_calibrated:
            return np.zeros(1, dtype=np.float32), False
        wrench = np.asarray(ati_window, dtype=np.float64)
        if wrench.ndim != 2 or wrench.shape[1] != 6:
            raise ValueError("ati_window must have shape [L,6]")
        transform = np.asarray(calibration.T_ati_to_gripper, dtype=np.float64)
        axis = np.asarray(calibration.gripper_normal_axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        force = (wrench[:, :3] - np.asarray(calibration.bias_wrench[:3])) @ transform[:3, :3].T
        normal = calibration.force_sign * (force @ axis)
        return normal.astype(np.float32), True

    def _make_batch(
        self,
        gelsight_window: np.ndarray,
        ati_window: np.ndarray | None,
        gripper_state: Sequence[float] | np.ndarray,
        calibration: ForceCalibration | None,
        frame_timestamps_sec: np.ndarray | None,
        ati_timestamps_sec: np.ndarray | None,
    ) -> Any:
        import torch

        frames = self._frames(gelsight_window)
        count = frames.shape[0]
        if frame_timestamps_sec is None:
            frame_times = np.arange(count, dtype=np.float32) / 30.0
        else:
            raw_times = np.asarray(frame_timestamps_sec, dtype=np.float64)
            indices = np.linspace(0, raw_times.size - 1, count).round().astype(int)
            frame_times = (raw_times[indices] - raw_times[indices][0]).astype(np.float32)
        delta_t = np.diff(frame_times, prepend=frame_times[0]).astype(np.float32)
        normal, force_available = self._normal_force(ati_window, calibration)
        if force_available:
            high_times = (
                np.asarray(ati_timestamps_sec, dtype=np.float32)
                if ati_timestamps_sec is not None
                else np.arange(normal.size, dtype=np.float32) / 500.0
            )
            high_times -= high_times[0]
            frame_force = np.interp(frame_times, high_times, normal).astype(np.float32)
            scale = max(float(np.std(normal)), 1e-6)
            normalized = (frame_force - float(np.mean(normal))) / scale
        else:
            high_times = np.zeros(normal.shape, dtype=np.float32)
            frame_force = np.zeros(count, dtype=np.float32)
            normalized = np.zeros(count, dtype=np.float32)
        derivative = np.gradient(frame_force, frame_times, edge_order=1) if count > 1 else np.zeros_like(frame_force)
        tactile = torch.from_numpy(frames).unsqueeze(0)
        delta = torch.zeros_like(tactile)
        delta[:, 1:] = tactile[:, 1:] - tactile[:, :-1]
        batch_type = self._batch_type
        assert batch_type is not None
        return batch_type(
            fabric_id=["deployment"],
            observation_id=["window"],
            press_id=["press"],
            session_id=["offline"],
            split_name=["inference"],
            dataset_id=["fabric_droid"],
            rgb_image=torch.zeros(1, 3, 224, 224),
            rgb_available=torch.tensor([False]),
            rgb_source_type=["unavailable"],
            rgb_is_current_observation=torch.tensor([False]),
            rgb_side_metadata_available=torch.tensor([False]),
            tactile_raw=tactile,
            tactile_delta=delta,
            frame_timestamps_sec=torch.from_numpy(frame_times).unsqueeze(0),
            frame_delta_t_sec=torch.from_numpy(delta_t).unsqueeze(0),
            frame_valid_mask=torch.ones(1, count, dtype=torch.bool),
            sequence_length=torch.tensor([count]),
            canonical_normal_force_N=torch.from_numpy(frame_force).unsqueeze(0),
            normalized_normal_force=torch.from_numpy(normalized).unsqueeze(0),
            normal_force_derivative=torch.from_numpy(derivative.astype(np.float32)).unsqueeze(0),
            frame_force_valid_mask=torch.full((1, count), force_available, dtype=torch.bool),
            high_rate_normal_force=torch.from_numpy(normal).unsqueeze(0),
            high_rate_timestamps_sec=torch.from_numpy(high_times).unsqueeze(0),
            high_rate_force_mask=torch.full((1, normal.size), force_available, dtype=torch.bool),
            high_rate_force_available=torch.tensor([force_available]),
            reference_features=None,
            reference_valid_mask=torch.zeros(1, count, dtype=torch.bool),
            reference_available=torch.tensor([False]),
            reference_cell_id=[""],
            press_quality_score=torch.tensor([1.0]),
            quality_tier=["clean"],
            observed_fabric_side_target=["unknown"],
            fabric_side_label_available=torch.tensor([False]),
            fabric_side_label_confidence=torch.tensor([0.0]),
            garment_layer_target=["unknown"],
            garment_layer_label_available=torch.tensor([False]),
            opposite_side_available=torch.tensor([False]),
            task_text=[""],
            task_available=torch.tensor([False]),
            controller_phase_hint=None,
            controller_phase_available=None,
            auxiliary={"gripper_state": np.asarray(gripper_state).tolist()},
        ).to(self.device)

    def encode_policy_state(
        self,
        gelsight_window: np.ndarray,
        ati_window: np.ndarray | None = None,
        gripper_state: Sequence[float] | np.ndarray = (),
        *,
        calibration: ForceCalibration | None = None,
        frame_timestamps_sec: np.ndarray | None = None,
        ati_timestamps_sec: np.ndarray | None = None,
    ) -> PhysicalState:
        self._load()
        import torch

        batch = self._make_batch(
            gelsight_window,
            ati_window,
            gripper_state,
            calibration,
            frame_timestamps_sec,
            ati_timestamps_sec,
        )
        assert self._model is not None
        with torch.inference_mode():
            bundle, auxiliary = self._model(batch, return_auxiliary=True)
        stage1 = auxiliary["stage1"]
        legacy = auxiliary["legacy"]
        contact_logits = stage1["contact_logits"][0]
        contact_probability = float(torch.softmax(contact_logits, dim=-1)[..., -1].mean().cpu())
        phase_logits = stage1["motion_state_logits"][0]
        contact_phase = int(torch.mode(torch.argmax(phase_logits, dim=-1)).values.cpu())
        predicted_force = float(stage1["fused_force_pred"][0, -1].reshape(-1)[0].cpu())
        area_curve = legacy.response_predictions["contact_area_curve"][0]
        flattening = legacy.response_predictions["flattening_index"][0]
        response_uncertainty = legacy.response_uncertainties["flattening_index"][0]
        config_payload = {
            "frames": 16,
            "image_size": 224,
            "normal_force_calibrated": bool(calibration and calibration.normal_force_calibrated),
            "semantics": "single_press_observed_surface_only",
        }
        config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()
        physical_valid = bool(bundle.observation_valid[0].cpu()) if bundle.observation_valid is not None else True
        force_valid = bool(calibration and calibration.normal_force_calibrated and ati_window is not None)
        return PhysicalState(
            global_tactile_latent=bundle.global_physical_token[0].detach().float().cpu().numpy(),
            predicted_normal_force=predicted_force,
            contact_probability=contact_probability,
            contact_phase=contact_phase,
            contact_area_proxy=float(area_curve.mean().cpu()),
            softness_score=float(flattening.cpu()),
            uncertainty=float(response_uncertainty.cpu()),
            tactile_valid=bool(np.isfinite(gelsight_window).all() and np.asarray(gelsight_window).shape[0] >= 2),
            force_valid=force_valid,
            physical_state_valid=physical_valid,
            checkpoint_sha256=self.checkpoint_sha256,
            config_sha256=config_hash,
            semantics={
                "softness_score": "flattening/compression response proxy; not true material softness",
                "contact_area_proxy": "model response-head proxy; not calibrated physical area",
                "predicted_normal_force": "model prediction; ATI raw wrench is never assumed to be gripper-normal",
            },
        )
