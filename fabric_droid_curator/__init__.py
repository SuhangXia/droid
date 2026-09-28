"""Fabric-DROID Dataset Curator.

The package deliberately treats captured episode directories as immutable.
All derived state is written below the configured curation root.
"""

from .config import CuratorConfig, load_config

__all__ = ["CuratorConfig", "load_config"]
__version__ = "0.1.0"
