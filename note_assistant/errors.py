from __future__ import annotations

from typing import Callable


class ErrorBus:
    """Singleton event bus for backend errors. Backends emit; UI subscribes."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[str, str, str], None]] = []

    def subscribe(self, callback: Callable[[str, str, str], None]) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, str, str], None]) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def emit(self, source: str, message: str, severity: str = "error") -> None:
        for cb in self._subscribers:
            try:
                cb(source, message, severity)
            except Exception:
                pass


error_bus = ErrorBus()
