"""Tool registry for LLM tool/function calling.

Adding a new tool is a single decorator call — see `tools/terminal.py` for
an example. Tools live in this package; importing the package registers
them automatically (each tool module is imported below).

Tools can also access an optional `ToolContext` via the `context` ContextVar
to ask the user for confirmation mid-execution (e.g. "command still running,
keep waiting?"). The app installs a context implementation around
`tools.execute()` calls. When no context is set (e.g. in tests), tools
should fall back to a safe default.
"""
from __future__ import annotations

import importlib
import inspect
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

Handler = Callable[..., Awaitable[str]]


class ToolContext(Protocol):
    """Lets a running tool ask the user a yes/no question."""

    async def confirm(self, title: str, body: str) -> bool: ...


# ContextVar so tools can pull the current context implicitly without
# needing it threaded through every parameter. Per-task per asyncio.
context: ContextVar[ToolContext | None] = ContextVar(
    "tools_context", default=None
)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema (object)
    handler: Handler
    # If True, the app prompts the user for approval before each call
    # (unless auto-approve mode is active). Mark this for tools that
    # mutate the user's machine, send network requests, etc.
    requires_approval: bool = False


_REGISTRY: dict[str, Tool] = {}


def tool(
    *,
    name: str,
    description: str,
    parameters: dict,
    requires_approval: bool = False,
) -> Callable[[Handler], Handler]:
    """Register an async function as an LLM-callable tool.

    `parameters` must be a JSON Schema object describing the function's
    arguments. The handler must be async and return a string (the tool's
    output as the model will see it). Set `requires_approval=True` for
    tools that should be gated by an explicit user OK.
    """

    def decorator(fn: Handler) -> Handler:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"Tool handler '{name}' must be async (use 'async def').")
        if name in _REGISTRY:
            raise ValueError(f"Tool '{name}' is already registered.")
        _REGISTRY[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=fn,
            requires_approval=requires_approval,
        )
        return fn

    return decorator


def all_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


async def execute(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool by name. Returns the tool's string output, or a
    formatted error string if the tool doesn't exist or raises."""
    spec = _REGISTRY.get(name)
    if spec is None:
        return f"Error: unknown tool '{name}'"
    try:
        result = await spec.handler(**(arguments or {}))
    except TypeError as e:
        return f"Error: bad arguments for '{name}': {e}"
    except Exception as e:  # tool's runtime failure is data, not a crash
        return f"Error: {type(e).__name__}: {e}"
    return result if isinstance(result, str) else str(result)


# Importing the modules below has the side effect of registering their
# tools via the @tool decorator. Add new tool modules to `_BUILTIN_TOOLS`.
_BUILTIN_TOOLS: tuple[str, ...] = ("terminal",)
for _name in _BUILTIN_TOOLS:
    importlib.import_module(f"{__name__}.{_name}")
