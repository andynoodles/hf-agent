from __future__ import annotations

import asyncio

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.suggester import SuggestFromList
from textual.widgets import Footer, Header, Input, Static

from . import providers, tools
from .approval import ApprovalDecision, ApprovalScreen
from .command_input import CommandInput
from .commands import COMMAND_NAMES, SLASH_COMMANDS, matching
from .config import ModelChoice, available_models
from .confirm import ConfirmScreen
from .message_view import MessageView
from .model_select import ModelSelectScreen
from .providers import TextDelta, ToolCall
from .spinner import Spinner


class _AppToolContext:
    """Bridges running tools to the TUI for mid-execution prompts."""

    def __init__(self, app: "ChatApp") -> None:
        self._app = app

    async def confirm(self, title: str, body: str) -> bool:
        return await self._app.push_screen_wait(ConfirmScreen(title, body))

_MAX_TOOL_ITERATIONS = 8  # safety cap on consecutive tool-call rounds
_TOOL_OUTPUT_PREVIEW_CHARS = 600


class ChatApp(App):
    CSS = """
    Screen { background: $surface; }
    #chat { height: 1fr; }
    #status {
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
    }
    #cmd-hints {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $boost;
    }
    #cmd-hints.-hidden { display: none; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.history: list[providers.Message] = []
        models = available_models()
        self.current_model: ModelChoice | None = models[0] if models else None
        self._streaming = False
        # When True, tools that mark `requires_approval=True` run without
        # an approval prompt. Toggled by /auto.
        self._auto_approve = False

    # --- layout ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat")
        yield Static("", id="status")
        yield Static("", id="cmd-hints", classes="-hidden", markup=True)
        yield CommandInput(
            placeholder="Type a message or /  (Tab to complete commands)",
            id="prompt-input",
            suggester=SuggestFromList(COMMAND_NAMES, case_sensitive=False),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "TUI Chat"
        self._refresh_status()
        self._log_system(
            "Welcome! Use [b]/models[/b] to switch model, "
            "[b]/clear[/b] to reset, [b]/quit[/b] to exit."
        )
        if not self.current_model:
            self._log_system(
                "[red]No models configured.[/] "
                "Set OPENAI_API_KEY/OPENAI_MODEL or GEMINI_API_KEY/GEMINI_MODEL in .env"
            )
        self.query_one("#prompt-input", Input).focus()

    # --- helpers ---------------------------------------------------------

    def _refresh_status(self) -> None:
        self._set_status_streaming(self._streaming)

    def _log_system(self, msg: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        view = MessageView("system")
        chat.mount(view)
        view.set_text(msg)
        chat.scroll_end(animate=False)

    # --- input -----------------------------------------------------------

    @on(Input.Changed, "#prompt-input")
    def _on_changed(self, event: Input.Changed) -> None:
        hints = self.query_one("#cmd-hints", Static)
        value = event.value
        if not value.startswith("/"):
            hints.add_class("-hidden")
            return
        matches = matching(value)
        if not matches:
            hints.update(f"  [dim]no command matches[/] [b]{escape(value)}[/]")
            hints.remove_class("-hidden")
            return
        rendered = "  ".join(
            f"[b]{name}[/] [dim]— {desc}[/]" for name, desc in matches
        )
        hints.update(f"  {rendered}")
        hints.remove_class("-hidden")

    @on(Input.Submitted, "#prompt-input")
    async def _on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if text.startswith("/"):
            self._handle_command(text)
            return

        if self._streaming:
            self._log_system("[yellow]Already streaming, please wait...[/]")
            return
        if not self.current_model:
            self._log_system("[red]No model selected. Use /models[/]")
            return

        # Hide hints once a real message is sent.
        self.query_one("#cmd-hints", Static).add_class("-hidden")

        self.history.append({"role": "user", "content": text})

        chat = self.query_one("#chat", VerticalScroll)
        user_view = MessageView("user")
        await chat.mount(user_view)
        user_view.set_text(text)
        chat.scroll_end(animate=False)

        self._stream_response()

    # --- slash commands --------------------------------------------------

    def _handle_command(self, text: str) -> None:
        cmd = text.split(maxsplit=1)[0].lower()
        if cmd not in SLASH_COMMANDS:
            self._log_system(f"[red]Unknown command:[/] {escape(cmd)}")
            return
        if cmd == "/quit":
            self.exit()
        elif cmd == "/clear":
            self._clear()
        elif cmd == "/models":
            self._select_model()
        elif cmd == "/auto":
            self._toggle_auto_approve()
        elif cmd == "/help":
            lines = [
                f"[b]{name}[/] [dim]— {desc}[/]"
                for name, desc in SLASH_COMMANDS.items()
            ]
            self._log_system("Commands:  " + "   ".join(lines))

    async def action_clear_chat(self) -> None:
        await self._clear_async()

    def _clear(self) -> None:
        self.run_worker(self._clear_async(), exclusive=False)

    async def _clear_async(self) -> None:
        self.history.clear()
        chat = self.query_one("#chat", VerticalScroll)
        await chat.remove_children()
        self._log_system("History cleared.")

    def _toggle_auto_approve(self) -> None:
        self._auto_approve = not self._auto_approve
        if self._auto_approve:
            self._log_system(
                "[yellow]⚠ Auto-approve ON[/] — tool calls run without prompting. "
                "Type [b]/auto[/] again to disable."
            )
        else:
            self._log_system("[green]Auto-approve OFF[/] — tool calls require approval.")
        self._refresh_status()

    def _select_model(self) -> None:
        choices = available_models()
        if not choices:
            self._log_system(
                "[red]No models configured.[/] "
                "Set OPENAI_API_KEY or GEMINI_API_KEY in .env"
            )
            return

        def _on_chosen(choice: ModelChoice | None) -> None:
            if choice:
                self.current_model = choice
                self._refresh_status()
                self._log_system(f"Switched to [b]{choice.label}[/b]")

        self.push_screen(ModelSelectScreen(choices), _on_chosen)

    # --- streaming -------------------------------------------------------

    @work(exclusive=True)
    async def _stream_response(self) -> None:
        assert self.current_model is not None
        self._streaming = True
        self._set_status_streaming(True)

        chat = self.query_one("#chat", VerticalScroll)
        active_tools = tools.all_tools()

        try:
            for _ in range(_MAX_TOOL_ITERATIONS):
                msg = MessageView("assistant", self.current_model.model)
                await chat.mount(msg)
                chat.scroll_end(animate=False)

                spinner = Spinner(self, lambda f, m=msg: m.set_text(f"_{f} thinking…_"))
                spinner.start()

                text_buf: list[str] = []
                tool_calls: list[ToolCall] = []

                try:
                    async for event in providers.stream(
                        self.current_model, self.history, active_tools
                    ):
                        spinner.stop()  # idempotent; first event wins
                        if isinstance(event, TextDelta):
                            text_buf.append(event.text)
                            msg.set_text("".join(text_buf))
                            chat.scroll_end(animate=False)
                            await asyncio.sleep(0)
                        elif isinstance(event, ToolCall):
                            tool_calls.append(event)
                finally:
                    spinner.stop()

                full_text = "".join(text_buf)
                if not full_text and not tool_calls:
                    msg.set_text("_(no content)_")
                elif not full_text:
                    # Hide the empty bubble when the turn was tool-only.
                    await msg.remove()

                # Record assistant turn (text + any tool requests).
                assistant_record: dict = {"role": "assistant", "content": full_text}
                if tool_calls:
                    assistant_record["tool_calls"] = [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in tool_calls
                    ]
                self.history.append(assistant_record)

                if not tool_calls:
                    break  # plain text turn — nothing more to do

                # Execute each tool, surface it in the chat, append to history.
                for tc in tool_calls:
                    await self._run_tool_call(tc, chat)
            else:
                self._log_system(
                    f"[yellow]Stopped: tool loop hit cap of "
                    f"{_MAX_TOOL_ITERATIONS} rounds.[/]"
                )
        except Exception as e:
            err = MessageView("system")
            await chat.mount(err)
            err.set_text(
                f"[red]Error from {self.current_model.provider}:[/] {escape(str(e))}"
            )
            chat.scroll_end(animate=False)
        finally:
            self._streaming = False
            self._set_status_streaming(False)

    async def _run_tool_call(self, tc: ToolCall, chat: VerticalScroll) -> None:
        spec = tools.get_tool(tc.name)
        args_preview = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())

        # Mount the call bubble first so it stays in chat order.
        call_view = MessageView("system")
        await chat.mount(call_view)
        call_view.set_text(
            f"[b]🔧 tool call[/] [cyan]{escape(tc.name)}[/]({escape(args_preview)})"
        )
        chat.scroll_end(animate=False)

        # Approval gate: tools that opt-in via `requires_approval` get a
        # modal unless the user has explicitly enabled auto-approve.
        needs_approval = bool(spec and spec.requires_approval) and not self._auto_approve
        if needs_approval:
            call_view.set_text(
                f"[b yellow]⏸ awaiting approval[/] [cyan]{escape(tc.name)}[/]"
                f"({escape(args_preview)})"
            )
            decision: ApprovalDecision = await self.push_screen_wait(
                ApprovalScreen(tc.name, tc.arguments)
            )
            if decision.enable_auto:
                self._auto_approve = True
                self._refresh_status()
                self._log_system(
                    "[yellow]⚠ Auto-approve ON[/] — future tool calls will run "
                    "without prompting. Type [b]/auto[/] to disable."
                )
            if not decision.approved:
                call_view.set_text(
                    f"[b red]🚫 tool denied by user[/] [cyan]{escape(tc.name)}[/]"
                    f"({escape(args_preview)})"
                )
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": "Tool execution denied by user.",
                    }
                )
                return
            # Approved — return the bubble to the running state.
            call_view.set_text(
                f"[b]🔧 tool call[/] [cyan]{escape(tc.name)}[/]({escape(args_preview)})"
            )

        # Install the tool context so the tool can prompt the user
        # mid-execution (e.g. "command still running, keep waiting?").
        token = tools.context.set(_AppToolContext(self))
        try:
            result = await tools.execute(tc.name, tc.arguments)
        finally:
            tools.context.reset(token)

        preview = result if len(result) <= _TOOL_OUTPUT_PREVIEW_CHARS else (
            result[:_TOOL_OUTPUT_PREVIEW_CHARS]
            + f"\n… [{len(result) - _TOOL_OUTPUT_PREVIEW_CHARS} more chars]"
        )
        result_view = MessageView("system")
        await chat.mount(result_view)
        result_view.set_text(
            f"[b]↪ tool result[/] [cyan]{escape(tc.name)}[/]\n"
            f"[dim]{escape(preview)}[/]"
        )
        chat.scroll_end(animate=False)

        self.history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": result,
            }
        )

    def _set_status_streaming(self, streaming: bool) -> None:
        status = self.query_one("#status", Static)
        if not self.current_model:
            status.update(" [red]No model[/red]")
            return
        parts = [f" Model: [b]{self.current_model.label}[/b]"]
        if self._auto_approve:
            parts.append("[yellow]auto-approve[/]")
        else:
            parts.append("[dim]approval required[/]")
        if streaming:
            parts.append("[yellow]● streaming…[/]")
        status.update("  ".join(parts))
