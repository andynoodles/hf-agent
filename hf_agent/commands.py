from __future__ import annotations

# Single source of truth for slash commands. The app uses this for both
# the typed-command dispatcher and the autocomplete UI, so the two never
# drift apart.
SLASH_COMMANDS: dict[str, str] = {
    "/models": "switch model",
    "/auto": "toggle auto-approve for tool calls",
    "/clear": "reset history",
    "/help": "show commands",
    "/quit": "exit",
}

COMMAND_NAMES: tuple[str, ...] = tuple(SLASH_COMMANDS.keys())


def matching(prefix: str) -> list[tuple[str, str]]:
    """Return (command, description) pairs whose name starts with `prefix`.
    Case-insensitive. Empty/non-slash prefix returns no matches."""
    if not prefix.startswith("/"):
        return []
    p = prefix.lower()
    return [(k, v) for k, v in SLASH_COMMANDS.items() if k.startswith(p)]


def longest_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    s1 = min(strings)
    s2 = max(strings)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1
