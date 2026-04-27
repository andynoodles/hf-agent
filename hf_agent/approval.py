from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    enable_auto: bool = False  # if True, also flip the app into auto-approve mode

    @classmethod
    def deny(cls) -> "ApprovalDecision":
        return cls(approved=False, enable_auto=False)

    @classmethod
    def approve_once(cls) -> "ApprovalDecision":
        return cls(approved=True, enable_auto=False)

    @classmethod
    def approve_always(cls) -> "ApprovalDecision":
        return cls(approved=True, enable_auto=True)


def _format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


class ApprovalScreen(ModalScreen[ApprovalDecision]):
    """Modal asking the user to approve a tool call.

    Keys: y / enter = approve once, a = approve + auto-approve mode,
    n / escape = deny.
    """

    BINDINGS = [
        Binding("y,enter", "approve_once", "Approve"),
        Binding("a", "approve_always", "Approve + auto"),
        Binding("n,escape", "deny", "Deny"),
    ]

    DEFAULT_CSS = """
    ApprovalScreen { align: center middle; }
    #approval-dialog {
        width: 80;
        height: auto;
        max-height: 80%;
        border: round $warning;
        padding: 1 2;
        background: $surface;
    }
    #approval-title {
        content-align: center middle;
        height: 1;
        color: $warning;
        margin-bottom: 1;
    }
    #approval-call {
        height: auto;
        padding: 1;
        background: $boost;
        margin-bottom: 1;
    }
    #approval-buttons { height: auto; align-horizontal: center; }
    #approval-buttons Button { margin: 0 1; }
    #approval-hint {
        height: 1;
        content-align: center middle;
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.arguments = arguments

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="approval-dialog"):
            yield Static("⚠  Tool approval required", id="approval-title")
            yield Static(
                f"[b cyan]{escape(self.tool_name)}[/]("
                f"{escape(_format_args(self.arguments))})",
                id="approval-call",
                markup=True,
            )
            with Horizontal(id="approval-buttons"):
                yield Button("Approve (y)", id="btn-approve", variant="success")
                yield Button("Approve + auto (a)", id="btn-auto", variant="primary")
                yield Button("Deny (n)", id="btn-deny", variant="error")
            yield Static(
                "[dim]y = approve once · a = approve and stop asking · n / esc = deny[/]",
                id="approval-hint",
                markup=True,
            )

    def on_mount(self) -> None:
        self.query_one("#btn-approve", Button).focus()

    # --- button handlers ---

    @on(Button.Pressed, "#btn-approve")
    def _btn_approve(self) -> None:
        self.dismiss(ApprovalDecision.approve_once())

    @on(Button.Pressed, "#btn-auto")
    def _btn_auto(self) -> None:
        self.dismiss(ApprovalDecision.approve_always())

    @on(Button.Pressed, "#btn-deny")
    def _btn_deny(self) -> None:
        self.dismiss(ApprovalDecision.deny())

    # --- key bindings ---

    def action_approve_once(self) -> None:
        self.dismiss(ApprovalDecision.approve_once())

    def action_approve_always(self) -> None:
        self.dismiss(ApprovalDecision.approve_always())

    def action_deny(self) -> None:
        self.dismiss(ApprovalDecision.deny())
