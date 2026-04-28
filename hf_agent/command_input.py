from __future__ import annotations

from textual.binding import Binding
from textual.widgets import Input

from .commands import command_names, matching


def _longest_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


class CommandInput(Input):
    """Input with Tab-complete for slash commands.

    - When the value starts with `/`, Tab completes to the longest common
      prefix of the matching commands. If only one match remains, Tab
      completes to the full command name.
    - When the value does not start with `/`, Tab is a no-op so the user
      can keep typing without surprising focus jumps.
    The inline ghost-text suggestion (right arrow accepts) is provided
    separately by an `Input.suggester`; Tab is the primary completion key.
    """

    BINDINGS = [Binding("tab", "complete", "Complete", show=False)]

    async def action_complete(self) -> None:
        value = self.value
        if not value.startswith("/"):
            return

        names = [name for name, _ in matching(value)]
        if not names:
            return

        target = names[0] if len(names) == 1 else _longest_common_prefix(names)
        if target and target != value:
            self.value = target
            self.cursor_position = len(target)
            return

        # If the value already equals the LCP and there are still multiple
        # candidates, fall through to whatever the suggester offered.
        if self._suggestion and self.value != self._suggestion:
            self.value = self._suggestion
            self.cursor_position = len(self._suggestion)


# Keep the names list importable from one place.
__all__ = ["CommandInput", "command_names"]
