"""Stable camera discovery for the Fabric-DROID collection UI."""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraDevice:
    kind: str
    serial: str
    name: str
    source: str
    physical_port: str = ""

    def label(self) -> str:
        port = f" | {self.physical_port}" if self.physical_port else ""
        return f"{self.name} | S/N {self.serial}{port}"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceInventory:
    realsense: tuple[CameraDevice, ...]
    uvc: tuple[CameraDevice, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "realsense": [device.to_dict() for device in self.realsense],
            "uvc": [device.to_dict() for device in self.uvc],
            "warnings": list(self.warnings),
        }


KNOWN_GELSIGHT_SERIALS = frozenset({"2DWF0RJM"})


def is_gelsight_device(device: CameraDevice) -> bool:
    """Return whether udev identity strongly identifies a GelSight camera."""

    identity = " ".join(
        (device.name, device.serial, device.source, device.physical_port)
    ).lower()
    hardware_serial = device.serial.split("@", 1)[0].upper()
    return hardware_serial in KNOWN_GELSIGHT_SERIALS or "gelsight" in identity


def is_integrated_webcam_device(device: CameraDevice) -> bool:
    """Return whether a UVC identity looks like the laptop's built-in webcam."""

    identity = " ".join(
        (device.name, device.source, device.physical_port)
    ).lower()
    return any(
        marker in identity
        for marker in ("hd webcam", "integrated camera", "integrated webcam", "chicony")
    )


def preferred_wrist_uvc_serial(
    devices: tuple[CameraDevice, ...] | list[CameraDevice],
    remembered_serial: str | None,
) -> str | None:
    """Prefer a non-GelSight external UVC camera for the wrist view."""

    candidates = [device for device in devices if not is_gelsight_device(device)]
    if remembered_serial and any(
        device.serial == remembered_serial for device in candidates
    ):
        return remembered_serial
    external = [
        device for device in candidates if not is_integrated_webcam_device(device)
    ]
    selected = external or candidates
    if not selected:
        return None
    return sorted(
        selected,
        key=lambda device: (device.name, device.serial, device.physical_port),
    )[0].serial


def preferred_gelsight_serial(
    devices: tuple[CameraDevice, ...] | list[CameraDevice],
    remembered_serial: str | None,
) -> str | None:
    """Prefer a positively identified GelSight over a stale webcam profile."""

    candidates = [device for device in devices if is_gelsight_device(device)]
    if candidates:
        if remembered_serial and any(
            device.serial == remembered_serial for device in candidates
        ):
            return remembered_serial
        return sorted(candidates, key=lambda device: (device.name, device.serial))[0].serial
    if remembered_serial and any(
        device.serial == remembered_serial for device in devices
    ):
        return remembered_serial
    return devices[0].serial if devices else None


def _udev_properties(path: Path) -> dict[str, str]:
    result = subprocess.run(
        ["udevadm", "info", "--query=property", f"--name={path}"],
        check=False,
        capture_output=True,
        text=True,
        timeout=2,
    )
    if result.returncode != 0:
        return {}
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    return properties


def _discover_realsense_python() -> list[CameraDevice]:
    import pyrealsense2 as rs

    devices: list[CameraDevice] = []
    for device in rs.context().devices:
        def info(field: Any) -> str:
            return device.get_info(field) if device.supports(field) else ""

        serial = info(rs.camera_info.serial_number)
        if not serial:
            continue
        devices.append(
            CameraDevice(
                kind="d435",
                serial=serial,
                name=info(rs.camera_info.name) or "Intel RealSense",
                source=serial,
                physical_port=info(rs.camera_info.physical_port),
            )
        )
    return devices


def _discover_video_nodes(video_root: Path) -> tuple[list[CameraDevice], list[CameraDevice]]:
    realsense: dict[str, CameraDevice] = {}
    uvc: dict[str, CameraDevice] = {}
    by_id_root = video_root / "v4l" / "by-id"
    stable_index_zero: dict[str, str] = {}
    if by_id_root.is_dir():
        for path in sorted(by_id_root.glob("*-video-index0")):
            try:
                target = str(path.resolve())
            except OSError:
                continue
            stable_index_zero[target] = str(path)

    for path in sorted(video_root.glob("video*")):
        properties = _udev_properties(path)
        if not properties:
            continue
        serial = properties.get("ID_SERIAL_SHORT") or properties.get("ID_SERIAL") or path.name
        product = (
            properties.get("ID_V4L_PRODUCT")
            or properties.get("ID_MODEL_FROM_DATABASE")
            or properties.get("ID_MODEL")
            or "UVC camera"
        ).replace("_", " ")
        physical_port = properties.get("ID_PATH", "")
        lowered = f"{product} {properties.get('ID_VENDOR', '')}".lower()
        if "realsense" in lowered:
            realsense.setdefault(
                serial,
                CameraDevice("d435", serial, product, serial, physical_port),
            )
            continue
        source = stable_index_zero.get(str(path.resolve()))
        if source is None:
            # Multi-interface cameras expose metadata-only nodes. Index zero is
            # the stable capture node when a by-id link is available.
            capabilities = properties.get("ID_V4L_CAPABILITIES", "")
            if ":capture:" not in capabilities:
                continue
            source = str(path)
        identity = serial
        existing = uvc.get(identity)
        if existing is not None and existing.physical_port != physical_port:
            # Cheap UVC devices often ship with the same placeholder serial
            # (for example, 200901010001). Preserve both cameras by extending
            # the UI identity with the stable physical USB port.
            suffix = _physical_port_key(physical_port) or path.name
            identity = f"{serial}@{suffix}"
            collision_index = 2
            while identity in uvc:
                identity = f"{serial}@{suffix}.{collision_index}"
                collision_index += 1
        uvc.setdefault(
            identity,
            CameraDevice("uvc", identity, product, source, physical_port),
        )
    return list(realsense.values()), list(uvc.values())


def _physical_port_key(value: str) -> str:
    """Normalize SDK sysfs and udev ID_PATH strings to the same USB port."""

    udev_match = re.search(r"usb-\d+:([0-9.]+):\d+\.\d+$", value)
    if udev_match:
        return udev_match.group(1)
    sysfs_matches = re.findall(r"/\d+-([0-9.]+)(?:/|:)", value)
    return sysfs_matches[-1] if sysfs_matches else ""


def discover_devices(video_root: Path = Path("/dev")) -> DeviceInventory:
    """Discover cameras without opening streams or changing camera controls."""

    warnings: list[str] = []
    python_realsense: list[CameraDevice] = []
    try:
        python_realsense = _discover_realsense_python()
    except Exception as exc:
        warnings.append(f"RealSense SDK discovery failed, using udev fallback: {exc}")
    udev_realsense, uvc = _discover_video_nodes(video_root)
    # A D435 can expose a V4L subdevice serial that differs from the serial
    # accepted by rs.config.enable_device(). Merge by physical USB port and
    # always prefer the RealSense SDK record.
    sdk_ports = {
        _physical_port_key(device.physical_port)
        for device in python_realsense
        if _physical_port_key(device.physical_port)
    }
    realsense = {
        device.serial: device
        for device in udev_realsense
        if not (
            _physical_port_key(device.physical_port)
            and _physical_port_key(device.physical_port) in sdk_ports
        )
    }
    realsense.update({device.serial: device for device in python_realsense})
    return DeviceInventory(
        realsense=tuple(sorted(realsense.values(), key=lambda device: device.serial)),
        uvc=tuple(sorted(uvc, key=lambda device: (device.name, device.serial))),
        warnings=tuple(warnings),
    )
