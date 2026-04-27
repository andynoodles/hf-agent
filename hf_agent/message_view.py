from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, Markdown, Static


class MessageView(Vertical):
    """A single chat message bubble.

    Assistant replies render as live Markdown — headings, lists, fenced
    code, and inline code re-render progressively as chunks stream in.
    User text is escaped (so `[..]` in input doesn't get parsed as Rich
    markup); system text is trusted markup since the app owns it.
    """

    DEFAULT_CSS = """
    MessageView { margin: 0 1 1 1; padding: 0 1; height: auto; }
    MessageView.role-user { border-left: thick $success; }
    MessageView.role-assistant { border-left: thick $accent; }
    MessageView.role-system { border-left: thick $warning; color: $text-muted; }
    MessageView > .msg-header { height: 1; padding: 0; }
    MessageView > .msg-body { height: auto; padding: 0; }
    MessageView Markdown { background: transparent; margin: 0; padding: 0; }
    """

    _PLACEHOLDER_MARKUP = "[dim]…[/]"
    _PLACEHOLDER_MD = "_…_"

    def __init__(self, role: str, model: str | None = None) -> None:
        super().__init__()
        self.role = role
        self.model = model
        self._text = ""
        self.add_class(f"role-{role}")

    def compose(self) -> ComposeResult:
        yield Label(self._header(), markup=True, classes="msg-header")
        if self.role == "assistant":
            yield Markdown(self._text or self._PLACEHOLDER_MD, classes="msg-body")
        else:
            yield Static(self._static_body(), markup=True, classes="msg-body")

    def _header(self) -> str:
        if self.role == "user":
            return "[bold green]You[/]"
        if self.role == "system":
            return "[bold yellow]System[/]"
        return f"[bold magenta]{self.model or 'assistant'}[/]"

    def _static_body(self) -> str:
        if not self._text:
            return self._PLACEHOLDER_MARKUP
        return self._text if self.role == "system" else escape(self._text)

    def set_text(self, text: str) -> None:
        """Replace the body text. Renders as markdown for assistant,
        as Rich markup for system, and as escaped plain text for user.
        Safe to call before compose runs — compose() reads self._text."""
        self._text = text
        if not self.children:
            return  # compose() will pick up self._text on first render
        if self.role == "assistant":
            self.query_one(Markdown).update(text or self._PLACEHOLDER_MD)
        else:
            self.query_one(".msg-body", Static).update(self._static_body())
