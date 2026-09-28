"""Thread-safe sensor lifecycle primitives."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any


class SensorStream(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"{self.name} is already started")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._guarded_run, name=f"fabric-droid-{self.name}", daemon=True)
        self._thread.start()

    def _guarded_run(self) -> None:
        try:
            self.run()
        except Exception as exc:  # the owner surfaces this during health/close
            self.error = exc
            self._stop_event.set()

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    def request_stop(self) -> None:
        """Ask the worker to stop without touching its backend cross-thread."""

        self._stop_event.set()

    def stop(self, timeout: float = 5.0) -> None:
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError(f"{self.name} did not stop within {timeout:.1f}s")
        self._thread = None
        if self.error is not None:
            raise RuntimeError(f"{self.name} failed") from self.error

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError
