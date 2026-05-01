"""Score a single tool call against a Case's Expect.

Returns a `Score` dataclass with `passed: bool` plus a human-readable
reason so failures are easy to triage when iterating prompts.

The DSL is intentionally small: bare value, any_of, present, absent,
contains, startswith, regex. See evals/cases.py for the full spec.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .cases import Case


@dataclass
class Score:
    case_id: str
    passed: bool
    reason: str
    actual_tool: str | None
    actual_args: dict[str, Any]


def _check_predicate(value: Any, pred: Any) -> tuple[bool, str]:
    """Apply a predicate to a single arg value. Returns (ok, reason)."""
    # Bare value → exact equality (with a small int/str leniency for
    # numeric args like `limit` that some providers stringify).
    if not isinstance(pred, tuple):
        if value == pred:
            return True, ""
        if isinstance(pred, (int, float)) and isinstance(value, str):
            try:
                if type(pred)(value) == pred:
                    return True, ""
            except (TypeError, ValueError):
                pass
        if isinstance(pred, str) and isinstance(value, (int, float)):
            if str(value) == pred:
                return True, ""
        return False, f"expected {pred!r}, got {value!r}"

    tag, *rest = pred
    if tag == "any_of":
        if value in rest:
            return True, ""
        return False, f"expected any of {rest!r}, got {value!r}"
    if tag == "present":
        return True, ""  # presence already verified by caller
    if tag == "absent":
        # Caller passes the value (None or sentinel) — handled outside.
        return False, "predicate 'absent' must be checked before reading the value"
    if tag == "contains":
        sub = rest[0]
        if isinstance(value, str) and sub.lower() in value.lower():
            return True, ""
        return False, f"expected contains {sub!r}, got {value!r}"
    if tag == "startswith":
        prefix = rest[0]
        if isinstance(value, str) and value.lower().startswith(prefix.lower()):
            return True, ""
        return False, f"expected startswith {prefix!r}, got {value!r}"
    if tag == "regex":
        pat = rest[0]
        if isinstance(value, str) and re.search(pat, value):
            return True, ""
        return False, f"expected regex {pat!r} to match, got {value!r}"
    return False, f"unknown predicate {pred!r}"


def score(case: Case, tool_name: str | None, arguments: dict[str, Any] | None) -> Score:
    expect = case.expect
    args = arguments or {}

    if expect.expects_no_tool_call:
        if tool_name is None:
            return Score(case.id, True, "no tool call (expected)", None, {})
        if tool_name in (expect.forbidden or ()):
            return Score(case.id, False,
                         f"called forbidden tool {tool_name!r}", tool_name, args)
        return Score(case.id, False,
                     f"emitted tool call {tool_name!r}, expected none", tool_name, args)

    if tool_name is None:
        if expect.accepts_no_call:
            return Score(case.id, True, "no tool call (accepted as defensive refusal)", None, {})
        return Score(case.id, False, "no tool call emitted", None, {})

    if tool_name in (expect.forbidden or ()):
        return Score(case.id, False,
                     f"called forbidden tool {tool_name!r}", tool_name, args)

    allowed_tools = list(expect.tool_any_of) or ([expect.tool] if expect.tool else [])
    if allowed_tools and tool_name not in allowed_tools:
        return Score(case.id, False,
                     f"wrong tool: got {tool_name!r}, expected one of {allowed_tools!r}",
                     tool_name, args)

    def _check_shape(shape: dict[str, Any]) -> tuple[bool, str]:
        for arg_name, pred in shape.items():
            if isinstance(pred, tuple) and pred and pred[0] == "absent":
                if arg_name in args:
                    return False, f"arg {arg_name!r} should be absent, got {args[arg_name]!r}"
                continue
            if arg_name not in args:
                return False, f"missing required arg {arg_name!r}"
            ok, why = _check_predicate(args[arg_name], pred)
            if not ok:
                return False, f"arg {arg_name!r}: {why}"
        return True, ""

    ok, why = _check_shape(expect.required or {})
    if not ok:
        return Score(case.id, False, why, tool_name, args)

    if expect.any_of_shapes:
        misses = []
        for shape in expect.any_of_shapes:
            ok, why = _check_shape(shape)
            if ok:
                return Score(case.id, True, "ok", tool_name, args)
            misses.append(why)
        return Score(case.id, False,
                     f"no any_of_shapes matched: [{'; '.join(misses)}]",
                     tool_name, args)

    return Score(case.id, True, "ok", tool_name, args)
