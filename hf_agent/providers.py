"""Provider streaming with tool-calling support.

Both providers yield a uniform event stream of:

  - TextDelta(text)        — text fragments as the model writes them
  - ToolCall(id, name, arguments)
                           — the model has requested to invoke a tool

The app accumulates events, then if any ToolCall was emitted, executes
the tools, appends their results to the history, and re-streams. The
generic history schema is:

  {"role": "system",    "content": str}
  {"role": "user",      "content": str}
  {"role": "assistant", "content": str, "tool_calls": [...]}   # tool_calls optional
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Each provider translates this generic format into its API-specific shape.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Union

from .config import ModelChoice
from .tools import Tool

Message = dict[str, Any]


# --- Stream events -----------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


StreamEvent = Union[TextDelta, ToolCall]


# --- OpenAI ------------------------------------------------------------


def _openai_history(history: list[Message]) -> list[dict]:
    """Convert generic history to OpenAI's chat-completions message list."""
    out: list[dict] = []
    for m in history:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": m.get("content") or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments") or {}),
                            },
                        }
                        for tc in m["tool_calls"]
                    ],
                }
            )
        elif role == "tool":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"],
                }
            )
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


def _openai_tools_payload(tools: list[Tool]) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


async def _stream_openai(
    model: str, history: list[Message], tools: list[Tool], allow_tools: bool
) -> AsyncIterator[StreamEvent]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _openai_history(history),
        "stream": True,
    }
    payload = _openai_tools_payload(tools)
    if payload and allow_tools:
        kwargs["tools"] = payload
    elif payload and not allow_tools:
        # Tools registered but caller wants a text-only turn (used to break
        # tool-call dead loops). Pass tools so the schema validates against
        # any prior tool messages, but forbid new calls.
        kwargs["tools"] = payload
        kwargs["tool_choice"] = "none"

    stream = await client.chat.completions.create(**kwargs)

    # OpenAI streams tool calls as deltas indexed by position; we
    # accumulate them and emit completed ToolCall events after the stream.
    pending: dict[int, dict[str, str]] = {}

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            yield TextDelta(delta.content)
        for tc_delta in getattr(delta, "tool_calls", None) or []:
            idx = tc_delta.index
            slot = pending.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc_delta.id:
                slot["id"] = tc_delta.id
            fn = getattr(tc_delta, "function", None)
            if fn:
                if fn.name:
                    slot["name"] = fn.name
                if fn.arguments:
                    slot["arguments"] += fn.arguments

    for slot in pending.values():
        try:
            args = json.loads(slot["arguments"]) if slot["arguments"] else {}
        except json.JSONDecodeError:
            args = {}
        yield ToolCall(id=slot["id"], name=slot["name"], arguments=args)


# --- Gemini ------------------------------------------------------------


def _gemini_contents(history: list[Message]):
    """Convert generic history to Gemini's `Content` list (excluding system)."""
    from google.genai import types

    contents: list[types.Content] = []
    for m in history:
        role = m["role"]
        if role == "system":
            continue
        if role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=m["content"])],
                )
            )
        elif role == "assistant":
            parts: list[types.Part] = []
            if m.get("content"):
                parts.append(types.Part.from_text(text=m["content"]))
            for tc in m.get("tool_calls") or []:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tc["name"],
                            args=tc.get("arguments") or {},
                        )
                    )
                )
            if parts:
                contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=m.get("name", ""),
                                response={"result": m["content"]},
                            )
                        )
                    ],
                )
            )
    return contents


def _gemini_system_instruction(history: list[Message]) -> str | None:
    for m in history:
        if m["role"] == "system":
            return m["content"]
    return None


# Gemini's FunctionDeclaration schema is a subset of OpenAPI; these keys
# are tolerated by OpenAI but cause Gemini to reject the whole tool set.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"additionalProperties", "$schema", "default", "examples", "exclusiveMinimum",
     "exclusiveMaximum", "patternProperties", "definitions", "$ref", "$defs"}
)


def _scrub_for_gemini(node):
    if isinstance(node, dict):
        return {
            k: _scrub_for_gemini(v)
            for k, v in node.items()
            if k not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(node, list):
        return [_scrub_for_gemini(x) for x in node]
    return node


def _gemini_tools_payload(tools: list[Tool]):
    from google.genai import types

    if not tools:
        return None
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters=_scrub_for_gemini(t.parameters),
                )
                for t in tools
            ]
        )
    ]


async def _stream_gemini(
    model: str, history: list[Message], tools: list[Tool], allow_tools: bool
) -> AsyncIterator[StreamEvent]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    tool_config = None
    if not allow_tools:
        # Force a text-only turn — used by the harness to break tool-call
        # dead loops. The model still sees the tool schema so prior
        # function_response parts validate, but cannot emit new calls.
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        )

    config = types.GenerateContentConfig(
        system_instruction=_gemini_system_instruction(history),
        tools=_gemini_tools_payload(tools),
        tool_config=tool_config,
        # Higher-entropy sampling. Gemini collapses to deterministic tool
        # spam at low temperature — these settings break it out of
        # repeated identical calls without sacrificing too much quality.
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        # Enable when using Gemini3 models
        # thinking_config=types.ThinkingConfig(thinking_level="medium"),
    )

    stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=_gemini_contents(history),
        config=config,
    )

    call_counter = 0
    async for chunk in stream:
        if chunk.text:
            yield TextDelta(chunk.text)

        # Each chunk's candidate may contain function_call parts. Gemini
        # generally emits them whole rather than streamed-in-pieces.
        for cand in getattr(chunk, "candidates", None) or []:
            content = getattr(cand, "content", None)
            if not content or not getattr(content, "parts", None):
                continue
            for part in content.parts:
                fc = getattr(part, "function_call", None)
                if not fc or not fc.name:
                    continue
                call_counter += 1
                call_id = getattr(fc, "id", None) or f"{fc.name}-{call_counter}"
                args = dict(fc.args) if getattr(fc, "args", None) else {}
                yield ToolCall(id=call_id, name=fc.name, arguments=args)


# --- Public API --------------------------------------------------------


def stream(
    choice: ModelChoice,
    history: list[Message],
    tools: list[Tool] | None = None,
    allow_tools: bool = True,
) -> AsyncIterator[StreamEvent]:
    """Stream a provider response.

    `allow_tools=False` forces the model to respond with text only — the
    harness uses this to break tool-call dead loops detected by the
    repetition guard.
    """
    tools = tools or []
    if choice.provider == "openai":
        return _stream_openai(choice.model, history, tools, allow_tools)
    if choice.provider == "gemini":
        return _stream_gemini(choice.model, history, tools, allow_tools)
    raise ValueError(f"Unknown provider: {choice.provider}")
