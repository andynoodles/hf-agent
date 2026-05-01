"""Non-interactive driver for the agent.

Runs one NL query through the same provider+tool loop the TUI uses,
streams a structured transcript to stdout, and exits. Used by the
`uv run main.py "<query>"` invocation and the break-it harness.

Tools that mutate the host (currently `run_shell`) are dropped from the
active tool set in headless mode — there is no human in the loop to
approve them.
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Callable

from . import doom_loop, prompts, providers, tools
from .config import ModelChoice, available_models
from .providers import TextDelta, ToolCall

_HEADLESS_TOOL_DENYLIST = frozenset({"run_shell"})
_DEFAULT_MAX_ROUNDS = 6
_TOOL_RESULT_PREVIEW_CHARS = 800

Emit = Callable[[dict], None]


def _resolve_model(spec: str | None) -> ModelChoice | None:
    """Pick a ModelChoice. Accepts 'model', 'provider:model', or label form."""
    choices = available_models()
    if not choices:
        return None
    if not spec:
        return choices[0]
    for c in choices:
        if spec in (c.model, f"{c.provider}:{c.model}", c.label):
            return c
    return None


def _ndjson_emit(out) -> Emit:
    def emit(rec: dict) -> None:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
    return emit


def _text_emit(out) -> Emit:
    def emit(rec: dict) -> None:
        ev = rec["event"]
        if ev == "user":
            out.write(f"\n=== USER (model: {rec['model']}) ===\n{rec['content']}\n")
        elif ev == "text":
            out.write(f"\n--- ASSISTANT (round {rec['round']}) ---\n{rec['content']}\n")
        elif ev == "tool_call":
            args = json.dumps(rec["arguments"], ensure_ascii=False)
            out.write(f"\n>>> TOOL CALL: {rec['name']}({args})\n")
        elif ev == "tool_result":
            out.write(f"<<< TOOL RESULT: {rec['name']}\n{rec['content']}\n")
        elif ev == "final":
            out.write(f"\n=== FINAL (round {rec['round']}) ===\n{rec['content']}\n")
        elif ev == "guard":
            out.write(f"\n!!! GUARD: {rec['content']}\n")
        elif ev == "stop":
            out.write(f"\n!!! STOPPED: {rec['reason']}\n")
        out.flush()
    return emit


async def run_once(
    query: str,
    *,
    model: str | None = None,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    as_json: bool = False,
    out=None,
) -> int:
    """Run one NL query through the tool loop. Returns shell-style exit code:
    0 = model produced a final text reply, 1 = stopped (guard / cap),
    2 = misconfiguration (no model)."""
    out = out if out is not None else sys.stdout
    choice = _resolve_model(model)
    if choice is None:
        sys.stderr.write(
            f"ERROR: no model configured matching {model!r}. "
            "Set OPENAI_API_KEY/OPENAI_MODEL or GEMINI_API_KEY/GEMINI_MODEL in .env.\n"
        )
        return 2

    history: list[dict[str, Any]] = [
        {"role": "system", "content": prompts.GROUNDING},
        {"role": "user", "content": query},
    ]
    active_tools = [t for t in tools.all_tools() if t.name not in _HEADLESS_TOOL_DENYLIST]

    emit = _ndjson_emit(out) if as_json else _text_emit(out)
    emit({"event": "user", "content": query, "model": choice.label})

    force_text_only = False
    last_round = -1
    for round_idx in range(max_rounds):
        last_round = round_idx
        nudge = doom_loop.check(history)
        if nudge:
            history.append({"role": "user", "content": nudge})
            force_text_only = True
            emit({"event": "guard", "content": nudge})

        text_buf: list[str] = []
        tool_calls: list[ToolCall] = []
        try:
            async for ev in providers.stream(
                choice, history, active_tools, allow_tools=not force_text_only
            ):
                if isinstance(ev, TextDelta):
                    text_buf.append(ev.text)
                elif isinstance(ev, ToolCall):
                    tool_calls.append(ev)
        except Exception as e:
            emit({"event": "stop", "reason": f"provider error: {type(e).__name__}: {e}"})
            return 1

        full_text = "".join(text_buf)
        if full_text:
            emit({"event": "text", "round": round_idx, "content": full_text})

        record: dict[str, Any] = {"role": "assistant", "content": full_text}
        if tool_calls:
            record["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in tool_calls
            ]
        history.append(record)

        if not tool_calls:
            emit({"event": "final", "round": round_idx, "content": full_text})
            return 0

        if force_text_only:
            emit({
                "event": "stop",
                "reason": "model emitted tool calls in forced text-only turn",
            })
            return 1

        for tc in tool_calls:
            emit({
                "event": "tool_call",
                "round": round_idx,
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            })
            result = await tools.execute(tc.name, tc.arguments)
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": result,
            })
            preview = (
                result if len(result) <= _TOOL_RESULT_PREVIEW_CHARS
                else result[:_TOOL_RESULT_PREVIEW_CHARS]
                     + f"... [{len(result) - _TOOL_RESULT_PREVIEW_CHARS} more chars]"
            )
            emit({
                "event": "tool_result",
                "round": round_idx,
                "id": tc.id,
                "name": tc.name,
                "content": preview,
            })

        force_text_only = False

    emit({
        "event": "stop",
        "reason": f"hit max-rounds cap ({max_rounds}) after round {last_round}",
    })
    return 1


def cli_run(
    query: str,
    *,
    model: str | None = None,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    as_json: bool = False,
) -> int:
    return asyncio.run(run_once(
        query, model=model, max_rounds=max_rounds, as_json=as_json
    ))
