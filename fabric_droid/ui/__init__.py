"""Desktop collection UI support for Fabric-DROID."""

from fabric_droid.ui.devices import CameraDevice, DeviceInventory, discover_devices
from fabric_droid.ui.episodes import EpisodeSummary, discover_episodes

__all__ = [
    "CameraDevice",
    "DeviceInventory",
    "EpisodeSummary",
    "discover_devices",
    "discover_episodes",
]
