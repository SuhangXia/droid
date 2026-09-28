from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .readers import load_ati, load_robot, load_stream_timestamps


@dataclass
class SignalBundle:
    robot_timestamp_ns: np.ndarray
    joint_velocity_norm: np.ndarray
    eef_linear_velocity: np.ndarray
    eef_angular_velocity: np.ndarray
    gripper_position: np.ndarray
    gripper_target: np.ndarray
    gripper_velocity: np.ndarray
    ati_timestamp_ns: np.ndarray
    ati_wrench: np.ndarray
    ati_force_delta: np.ndarray
    ati_derivative: np.ndarray
    gelsight_timestamp_ns: np.ndarray
    gelsight_image_delta: np.ndarray
    gelsight_contact_score: np.ndarray

    def to_npz(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(value) for name, value in vars(self).items()}


def _gradient(values: np.ndarray, timestamps_ns: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return np.zeros_like(values, dtype=np.float64)
    seconds = timestamps_ns.astype(np.float64) / 1e9
    return np.gradient(values, seconds, axis=0, edge_order=1)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    if values.size == 0 or width <= 1:
        return values.astype(np.float64, copy=True)
    width = min(width, values.size)
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _gelsight_features(episode_dir: Path, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(episode_dir / "tactile" / "gelsight_left.mp4"))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (96, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
    capture.release()
    count = min(len(frames), int(timestamps.size))
    if not count:
        empty = np.asarray([], dtype=np.float64)
        return timestamps[:0], empty, empty
    stack = np.stack(frames[:count])
    baseline_count = max(1, min(20, count // 10 if count >= 10 else count))
    baseline = np.median(stack[:baseline_count], axis=0)
    adjacent = np.zeros(count, dtype=np.float64)
    if count > 1:
        adjacent[1:] = np.mean(np.abs(np.diff(stack, axis=0)), axis=(1, 2)) / 255.0
    contact = np.mean(np.abs(stack - baseline), axis=(1, 2)) / 255.0
    return timestamps[:count], _smooth(adjacent, 5), _smooth(contact, 7)


def compute_signals(episode_dir: Path) -> SignalBundle:
    robot = load_robot(episode_dir)
    robot_ns = robot["timestamp_ns"]
    joint_norm = np.linalg.norm(robot["joint_velocity"], axis=1)
    pose_velocity = _gradient(robot["eef_pose"], robot_ns)
    eef_linear = np.linalg.norm(pose_velocity[:, :3], axis=1)
    eef_angular = np.linalg.norm(pose_velocity[:, 3:6], axis=1)
    gripper = robot["gripper_position"].reshape(-1)
    gripper_target = robot["gripper_target"].reshape(-1)
    gripper_velocity = _gradient(gripper, robot_ns)

    ati = load_ati(episode_dir)
    ati_ns = ati["timestamp_ns"]
    wrench = np.column_stack([ati[name] for name in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")])
    if wrench.size:
        baseline_count = max(1, min(wrench.shape[0] // 10, 1000))
        baseline = np.median(wrench[:baseline_count], axis=0)
        force_delta = np.linalg.norm(wrench[:, :3] - baseline[:3], axis=1)
        ati_derivative = np.abs(_gradient(force_delta, ati_ns))
        force_delta = _smooth(force_delta, 25)
        ati_derivative = _smooth(ati_derivative, 25)
    else:
        force_delta = np.asarray([], dtype=np.float64)
        ati_derivative = np.asarray([], dtype=np.float64)

    gel_ns = load_stream_timestamps(episode_dir, "gelsight")
    gel_ns, gel_delta, gel_contact = _gelsight_features(episode_dir, gel_ns)
    return SignalBundle(
        robot_timestamp_ns=robot_ns,
        joint_velocity_norm=_smooth(joint_norm, 3),
        eef_linear_velocity=_smooth(eef_linear, 3),
        eef_angular_velocity=_smooth(eef_angular, 3),
        gripper_position=gripper,
        gripper_target=gripper_target,
        gripper_velocity=_smooth(gripper_velocity, 3),
        ati_timestamp_ns=ati_ns,
        ati_wrench=wrench,
        ati_force_delta=force_delta,
        ati_derivative=ati_derivative,
        gelsight_timestamp_ns=gel_ns,
        gelsight_image_delta=gel_delta,
        gelsight_contact_score=gel_contact,
    )


def save_signals(bundle: SignalBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **bundle.to_npz())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_signals(path: Path) -> SignalBundle:
    with np.load(path) as payload:
        return SignalBundle(**{name: np.asarray(payload[name]) for name in SignalBundle.__annotations__})


def get_or_compute_signals(episode_dir: Path, path: Path, *, force: bool = False) -> SignalBundle:
    if path.is_file() and not force:
        return load_signals(path)
    bundle = compute_signals(episode_dir)
    save_signals(bundle, path)
    return bundle


def downsample_signals(bundle: SignalBundle, *, max_points: int = 2000) -> dict[str, list[float | int]]:
    result: dict[str, list[float | int]] = {}

    def add_group(timestamp_name: str, names: tuple[str, ...]) -> None:
        timestamps = getattr(bundle, timestamp_name)
        step = max(1, int(np.ceil(timestamps.size / max_points)))
        result[timestamp_name] = timestamps[::step].astype(np.int64).tolist()
        for name in names:
            values = getattr(bundle, name)
            if values.ndim == 2:
                for column in range(values.shape[1]):
                    result[f"{name}_{column}"] = values[::step, column].astype(float).tolist()
            else:
                result[name] = values[::step].astype(float).tolist()

    add_group(
        "robot_timestamp_ns",
        (
            "joint_velocity_norm",
            "eef_linear_velocity",
            "eef_angular_velocity",
            "gripper_position",
            "gripper_target",
            "gripper_velocity",
        ),
    )
    add_group("ati_timestamp_ns", ("ati_wrench", "ati_force_delta", "ati_derivative"))
    add_group("gelsight_timestamp_ns", ("gelsight_image_delta", "gelsight_contact_score"))
    return result
