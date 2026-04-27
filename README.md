# tui-chat

A terminal chat UI for talking to LLMs over OpenAI-compatible and Google
Gemini endpoints. Built on [Textual](https://textual.textualize.io/).

- Streaming replies, rendered as live Markdown
- Slash commands with inline hints and Tab-completion
- Switch models on the fly with `/models`
- Tool / function calling with a one-line decorator
- A `run_shell` tool gated by an approval modal (or `/auto` to skip prompts)
- Animated thinking indicator while the model is working

## Setup

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
uv run main.py
```

## Configuration

Edit `.env`:

```ini
# OpenAI or any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, ...)
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
# OPENAI_MODELS=gpt-4o-mini,gpt-4o,gpt-4.1-mini   # comma-separated, optional

# Google Gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash
# GEMINI_MODELS=gemini-2.5-flash,gemini-2.0-flash
```

Either provider can be left blank — the app picks up whichever keys are
set. The `/models` picker lists every model from every configured
provider; the first one becomes the default at startup.

## Usage

Type a message and hit Enter. Slash commands:

| Command   | What it does                              |
| --------- | ----------------------------------------- |
| `/models` | Open the model picker                     |
| `/auto`   | Toggle auto-approve for tool calls        |
| `/clear`  | Clear chat history                        |
| `/help`   | List all commands                         |
| `/quit`   | Exit                                      |

While typing a slash command, matching commands are shown inline below
the chat. **Tab** completes to the longest common prefix (or the full
command if only one matches). The right-arrow / inline ghost suggestion
also accepts the suggestion.

Keybindings:

- `Ctrl+L` — clear chat
- `Ctrl+C` — quit
- `Esc` — close any modal
- `y` / `n` / `a` — accept / deny / approve-and-auto on tool prompts

## Tool calling

The model can request tools, the app executes them, and the result is
fed back so the model can use it. The bundled tool is `run_shell`, which
executes a command in `/bin/sh` and returns stdout/stderr/exit code.

**Approval flow.** By default, every shell call shows an approval modal:

- `y` — approve once
- `a` — approve and switch on auto-approve mode for the rest of the
  session (also reachable via `/auto`)
- `n` / `Esc` — deny; the model is told the call was refused and can
  recover

If a command runs longer than its `timeout_seconds` (default 30), the
process is **not** killed automatically — instead, you get a
"⏳ command still running" prompt asking whether to keep waiting another
window. The command is only killed if you decline.

### Adding a new tool

Drop a module into `tui_chat/tools/` and decorate an async function:

```python
# tui_chat/tools/files.py
from . import tool

@tool(
    name="read_file",
    description="Read a UTF-8 text file and return its contents.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    requires_approval=False,  # set True to gate behind the approval modal
)
async def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()
```

Then add the module name to `_BUILTIN_TOOLS` in
`tui_chat/tools/__init__.py`. Both the OpenAI and Gemini providers will
automatically advertise the tool on the next call.

A tool can also ask the user a yes/no question mid-execution by reading
the current `ToolContext`:

```python
from . import context as tool_context, tool

@tool(...)
async def my_tool(...) -> str:
    ctx = tool_context.get()
    if ctx and not await ctx.confirm("Proceed?", "About to do something risky."):
        return "User declined."
    ...
```

## Project layout

```
.
├── main.py                       # entry point: load .env, run the app
├── requirements.txt
├── .env / .env.example
└── tui_chat/
    ├── app.py                    # ChatApp: layout, streaming + tool loop
    ├── config.py                 # ModelChoice, available_models()
    ├── providers.py              # OpenAI + Gemini streaming with tool events
    ├── commands.py               # SLASH_COMMANDS registry
    ├── command_input.py          # Input subclass with Tab-complete
    ├── message_view.py           # Per-turn message bubble (Markdown body)
    ├── model_select.py           # /models picker modal
    ├── approval.py               # Tool-call approval modal
    ├── confirm.py                # Generic yes/no modal
    ├── spinner.py                # Animated thinking spinner
    └── tools/
        ├── __init__.py           # Tool registry + @tool decorator + ToolContext
        └── terminal.py           # run_shell
```
