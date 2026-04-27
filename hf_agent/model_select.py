from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from .config import ModelChoice


class ModelSelectScreen(ModalScreen[ModelChoice | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    DEFAULT_CSS = """
    ModelSelectScreen { align: center middle; }
    #model-dialog {
        width: 60;
        height: auto;
        max-height: 80%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    #model-title {
        content-align: center middle;
        height: 1;
        color: $accent;
        margin-bottom: 1;
    }
    #model-list { height: auto; max-height: 20; }
    """

    def __init__(self, choices: list[ModelChoice]) -> None:
        super().__init__()
        self.choices = choices

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="model-dialog"):
            yield Static("Select a model (Esc to cancel)", id="model-title")
            yield OptionList(
                *[Option(c.label, id=str(i)) for i, c in enumerate(self.choices)],
                id="model-list",
            )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    @on(OptionList.OptionSelected)
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        idx = int(event.option.id)
        self.dismiss(self.choices[idx])
