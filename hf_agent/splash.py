from __future__ import annotations

from typing import Callable

from textual.containers import Vertical
from textual.widgets import Static


# ANSI Shadow block letters, hand-stitched so each row has a fixed
# width per glyph — the splash relies on column alignment.
_LETTERS: dict[str, tuple[str, ...]] = {
    "H": (
        "██╗  ██╗",
        "██║  ██║",
        "███████║",
        "██╔══██║",
        "██║  ██║",
        "╚═╝  ╚═╝",
    ),
    "F": (
        "███████╗",
        "██╔════╝",
        "█████╗  ",
        "██╔══╝  ",
        "██║     ",
        "╚═╝     ",
    ),
    "-": (
        "        ",
        "        ",
        "███████╗",
        "╚══════╝",
        "        ",
        "        ",
    ),
    "A": (
        " █████╗ ",
        "██╔══██╗",
        "███████║",
        "██╔══██║",
        "██║  ██║",
        "╚═╝  ╚═╝",
    ),
    "G": (
        " ██████╗ ",
        "██╔════╝ ",
        "██║  ███╗",
        "██║   ██║",
        "╚██████╔╝",
        " ╚═════╝ ",
    ),
    "E": (
        "███████╗",
        "██╔════╝",
        "█████╗  ",
        "██╔══╝  ",
        "███████╗",
        "╚══════╝",
    ),
    "N": (
        "███╗   ██╗",
        "████╗  ██║",
        "██╔██╗ ██║",
        "██║╚██╗██║",
        "██║ ╚████║",
        "╚═╝  ╚═══╝",
    ),
    "T": (
        "████████╗",
        "╚══██╔══╝",
        "   ██║   ",
        "   ██║   ",
        "   ██║   ",
        "   ╚═╝   ",
    ),
}

_TITLE = "HF-AGENT"
_ART_ROWS: tuple[str, ...] = tuple(
    "".join(_LETTERS[ch][row] for ch in _TITLE) for row in range(6)
)
_ART_WIDTH = len(_ART_ROWS[0])
_BYLINE = "made by andynoodles"

# Yellow → orange gradient cycled across the six rows.
_ROW_COLORS: tuple[str, ...] = (
    "#fff15a",
    "#ffd23d",
    "#ffa83d",
    "#ff7e3d",
    "#ff5a3d",
    "#ff3d3d",
)


class SplashScreen(Vertical):
    """Animated opening banner.

    Reveals the title row-by-row, types the byline beneath, holds
    briefly, then removes itself — leaving the chat scroll empty
    for the normal welcome message that follows.
    """

    DEFAULT_CSS = """
    SplashScreen {
        height: auto;
        padding: 1 2;
    }
    SplashScreen > Static {
        height: auto;
        width: 100%;
        text-align: center;
    }
    SplashScreen > #splash-byline {
        padding-top: 1;
    }
    """

    _ROW_DELAY = 0.05
    _AFTER_ART_PAUSE = 0.18
    _CHAR_DELAY = 0.025
    _HOLD = 0.7

    def __init__(self, on_done: Callable[[], None] | None = None) -> None:
        super().__init__(id="splash")
        self._on_done = on_done
        self._art = Static("", id="splash-art", markup=True)
        self._byline = Static("", id="splash-byline", markup=True)
        self._rows_shown = 0
        self._chars_typed = 0

    def compose(self):
        yield self._art
        yield self._byline

    def on_mount(self) -> None:
        # Paint the full-height placeholder once so the layout
        # doesn't reflow as rows reveal.
        self._render_art()
        self.set_timer(self._ROW_DELAY, self._reveal_next_row)

    # --- phase 1: reveal title rows top-to-bottom ----------------------
    def _reveal_next_row(self) -> None:
        self._rows_shown += 1
        self._render_art()
        if self._rows_shown < len(_ART_ROWS):
            self.set_timer(self._ROW_DELAY, self._reveal_next_row)
        else:
            self.set_timer(self._AFTER_ART_PAUSE, self._type_next_char)

    # --- phase 2: typewriter the byline --------------------------------
    def _type_next_char(self) -> None:
        self._chars_typed += 1
        self._render_byline()
        if self._chars_typed < len(_BYLINE):
            self.set_timer(self._CHAR_DELAY, self._type_next_char)
        else:
            self.set_timer(self._HOLD, self._dismiss)

    # --- phase 3: leave -------------------------------------------------
    def _dismiss(self) -> None:
        if self._on_done is not None:
            self._on_done()
        self.remove()

    # --- rendering ------------------------------------------------------
    def _render_art(self) -> None:
        lines: list[str] = []
        for i, row in enumerate(_ART_ROWS):
            if i < self._rows_shown:
                color = _ROW_COLORS[i % len(_ROW_COLORS)]
                lines.append(f"[{color}]{row}[/]")
            else:
                lines.append(" " * _ART_WIDTH)
        self._art.update("\n".join(lines))

    def _render_byline(self) -> None:
        if self._chars_typed == 0:
            self._byline.update("")
            return
        text = _BYLINE[: self._chars_typed]
        cursor = "▌" if self._chars_typed < len(_BYLINE) else " "
        self._byline.update(
            f"[#888888]──[/] [b #ffae3d]{text}[/][b #ffae3d]{cursor}[/] [#888888]──[/]"
        )
