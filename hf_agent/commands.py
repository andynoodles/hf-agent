from __future__ import annotations

from . import tools

# Static slash commands. Each registered tool also appears as a dynamic
# `/<tool_name>` toggle (see `slash_commands()` below) so users can opt
# tools into the session one at a time.
_BASE_COMMANDS: dict[str, str] = {
    "/models": "switch model",
    "/tool": "list available tools",
    "/loop": "/loop <goal> — run autonomous research/iteration loop",
    "/auto": "toggle auto-approve for tool calls",
    "/clear": "reset history",
    "/help": "show commands",
    "/quit": "exit",
}


def slash_commands() -> dict[str, str]:
    """Static commands plus a `/<tool_name>` toggle for each registered tool."""
    out = dict(_BASE_COMMANDS)
    for spec in tools.all_tools():
        out[f"/{spec.name}"] = f"/{spec.name} <request> — nudge model to use the '{spec.name}' tool"
    return out


def command_names() -> tuple[str, ...]:
    return tuple(slash_commands().keys())


def matching(prefix: str) -> list[tuple[str, str]]:
    """Return (command, description) pairs whose name starts with `prefix`.
    Case-insensitive. Empty/non-slash prefix returns no matches."""
    if not prefix.startswith("/"):
        return []
    p = prefix.lower()
    return [(k, v) for k, v in slash_commands().items() if k.startswith(p)]
