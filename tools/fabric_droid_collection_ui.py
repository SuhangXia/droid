#!/usr/bin/env python3
"""Launch the Fabric-DROID multimodal collection desktop UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# VS Code currently consumes the host's inotify watch budget. Avoid making
# Qt/IBus add another watch just to edit the UI's short ASCII metadata fields.
os.environ.setdefault("QT_IM_MODULE", "xim")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fabric_droid.ui.app import main


if __name__ == "__main__":
    raise SystemExit(main())
