"""Fabric-DROID sidecar collection and offline processing tools.

The package is deliberately independent from DROID's robot-control stack.
Importing it never connects to hardware or enables robot motion.
"""

from fabric_droid.schemas import EpisodeMetadata, EventMarker

__all__ = ["EpisodeMetadata", "EventMarker"]
__version__ = "0.1.0"
