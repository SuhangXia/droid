"""Timestamped Oculus reader adapter with a real stale-frame watchdog."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

from fabric_droid.robot.quest_teleop import QuestFrame


class TimestampedRightQuestSource:
    """Wrap the bundled Oculus APK stream and timestamp each fresh logcat frame."""

    def __init__(self, *, ip_address: Optional[str] = None) -> None:
        # Import only after the CLI has added the read-only submodule to sys.path.
        from oculus_reader.reader import OculusReader

        owner = self

        class TimestampedReader(OculusReader):
            def run(self) -> None:
                self.running = True
                self.device.shell(
                    'am start -n "com.rail.oculus.teleop/'
                    'com.rail.oculus.teleop.MainActivity" '
                    "-a android.intent.action.MAIN "
                    "-c android.intent.category.LAUNCHER"
                )
                self.thread = threading.Thread(
                    target=self.device.shell,
                    args=("logcat -T 0", self.read_logcat_by_line),
                    daemon=True,
                    name="fabric-droid-quest-logcat",
                )
                self.thread.start()

            def read_logcat_by_line(self, connection: Any) -> None:
                file_obj = connection.socket.makefile()
                try:
                    while self.running:
                        try:
                            line = file_obj.readline().strip()
                            data = self.extract_data(line)
                            if not data:
                                continue
                            transforms, buttons = self.process_data(data)
                            if transforms is None or buttons is None:
                                continue
                            with self._lock:
                                self.last_transforms = transforms
                                self.last_buttons = buttons
                                owner._record_fresh_frame(transforms, buttons)
                            if self.print_FPS:
                                self.fps_counter.getAndPrintFPS()
                        except UnicodeDecodeError:
                            continue
                finally:
                    file_obj.close()
                    connection.close()

            def stop(self) -> None:
                self.running = False
                thread = getattr(self, "thread", None)
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=1.0)

        self._frame_lock = threading.Lock()
        self._latest: Optional[QuestFrame] = None
        self._sequence = 0
        self._reader = TimestampedReader(ip_address=ip_address)

    def _record_fresh_frame(self, transforms: Any, buttons: Any) -> None:
        pose = transforms.get("r")
        if pose is None:
            return
        with self._frame_lock:
            self._sequence += 1
            self._latest = QuestFrame(
                pose=np.asarray(pose, dtype=np.float64).copy(),
                buttons=dict(buttons),
                received_monotonic_ns=time.monotonic_ns(),
                sequence=self._sequence,
            )

    def latest_frame(self) -> Optional[QuestFrame]:
        with self._frame_lock:
            if self._latest is None:
                return None
            return QuestFrame(
                pose=self._latest.pose.copy(),
                buttons=dict(self._latest.buttons),
                received_monotonic_ns=self._latest.received_monotonic_ns,
                sequence=self._latest.sequence,
            )

    def wait_for_frame(self, timeout_sec: float) -> QuestFrame:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            frame = self.latest_frame()
            if frame is not None:
                return frame
            time.sleep(0.02)
        raise TimeoutError(f"no right Quest controller frame received within {timeout_sec:.1f}s")

    def close(self) -> None:
        self._reader.stop()

    def __enter__(self) -> "TimestampedRightQuestSource":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

