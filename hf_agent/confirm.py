from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmScreen(ModalScreen[bool]):
    """Generic yes/no modal. Returns True for yes, False for no/escape."""

    BINDINGS = [
        Binding("y,enter", "yes", "Yes"),
        Binding("n,escape", "no", "No"),
    ]

    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    #confirm-dialog {
        width: 80;
        height: auto;
        max-height: 80%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    #confirm-title {
        content-align: center middle;
        height: 1;
        color: $accent;
        margin-bottom: 1;
    }
    #confirm-body {
        height: auto;
        padding: 1;
        background: $boost;
        margin-bottom: 1;
    }
    #confirm-buttons { height: auto; align-horizontal: center; }
    #confirm-buttons Button { margin: 0 1; }
    """

    def __init__(
        self,
        title: str,
        body: str,
        yes_label: str = "Yes (y)",
        no_label: str = "No (n)",
    ) -> None:
        super().__init__()
        self._title = title
        self._body = body
        self._yes_label = yes_label
        self._no_label = no_label

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="confirm-dialog"):
            yield Static(self._title, id="confirm-title", markup=True)
            yield Static(self._body, id="confirm-body", markup=True)
            with Horizontal(id="confirm-buttons"):
                yield Button(self._yes_label, id="btn-yes", variant="success")
                yield Button(self._no_label, id="btn-no", variant="error")

    def on_mount(self) -> None:
        self.query_one("#btn-yes", Button).focus()

    @on(Button.Pressed, "#btn-yes")
    def _btn_yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#btn-no")
    def _btn_no(self) -> None:
        self.dismiss(False)

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)
