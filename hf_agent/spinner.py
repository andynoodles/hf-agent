from __future__ import annotations

from typing import Callable

from textual.app import App
from textual.timer import Timer


class Spinner:
    """Animated spinner driven by Textual's `set_interval`.

    The caller supplies an `on_frame(frame)` callback so the spinner
    stays decoupled from any particular widget — the callback decides
    where to render the frame (Markdown body, Static, status bar, …).
    `start()` / `stop()` are idempotent so it's safe to call them from
    streaming hot paths.
    """

    FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    DEFAULT_INTERVAL = 0.08

    def __init__(
        self,
        app: App,
        on_frame: Callable[[str], None],
        interval: float = DEFAULT_INTERVAL,
    ) -> None:
        self._app = app
        self._on_frame = on_frame
        self._interval = interval
        self._idx = 0
        self._timer: Timer | None = None

    def start(self) -> None:
        if self._timer is not None:
            return
        self._tick()  # paint first frame immediately, no 80ms blank gap
        self._timer = self._app.set_interval(self._interval, self._tick)

    def stop(self) -> None:
        if self._timer is None:
            return
        self._timer.stop()
        self._timer = None

    def _tick(self) -> None:
        frame = self.FRAMES[self._idx]
        self._idx = (self._idx + 1) % len(self.FRAMES)
        self._on_frame(frame)
