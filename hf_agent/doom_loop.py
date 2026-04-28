"""Detect repeated tool-call patterns during autonomous loops.

A trimmed port of ml-intern's doom-loop detector. We hash recent tool
calls (and their results, when available) and look for two failure
modes: N identical consecutive calls, or a short repeating sequence
like A→B→A→B. When either fires, the caller should inject a corrective
system message so the model breaks the cycle.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]


@dataclass(frozen=True)
class _Sig:
    name: str
    args_hash: str
    result_hash: str | None


def _canon(args: Any) -> str:
    try:
        return json.dumps(args or {}, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(args)


def _hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _signatures(history: list[Message], lookback: int = 30) -> list[_Sig]:
    """Pull tool-call signatures from recent assistant turns, pairing
    each with the immediate tool result so polling (same args, changing
    results) doesn't trip the detector."""
    recent = history[-lookback:]
    sigs: list[_Sig] = []
    for i, msg in enumerate(recent):
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            result_hash: str | None = None
            for follow in recent[i + 1:]:
                role = follow.get("role")
                if role == "tool" and follow.get("tool_call_id") == tc.get("id"):
                    result_hash = _hash(str(follow.get("content", "")))
                    break
                if role in {"assistant", "user"}:
                    break
            sigs.append(
                _Sig(
                    name=tc.get("name", ""),
                    args_hash=_hash(_canon(tc.get("arguments"))),
                    result_hash=result_hash,
                )
            )
    return sigs


def _identical_consecutive(sigs: list[_Sig], threshold: int = 3) -> str | None:
    if len(sigs) < threshold:
        return None
    count = 1
    for i in range(1, len(sigs)):
        if sigs[i] == sigs[i - 1]:
            count += 1
            if count >= threshold:
                return sigs[i].name
        else:
            count = 1
    return None


def _repeating_sequence(sigs: list[_Sig]) -> list[_Sig] | None:
    n = len(sigs)
    for seq_len in range(2, 6):
        if n < seq_len * 2:
            continue
        pattern = sigs[-seq_len * 2 : -seq_len]
        reps = 0
        for start in range(n - seq_len, -1, -seq_len):
            if sigs[start : start + seq_len] == pattern:
                reps += 1
            else:
                break
        if reps >= 2:
            return pattern
    return None


def check(history: list[Message]) -> str | None:
    """Return a corrective prompt if the agent looks stuck, else None."""
    sigs = _signatures(history)
    if len(sigs) < 3:
        return None

    name = _identical_consecutive(sigs)
    if name:
        return (
            f"[SYSTEM: REPETITION GUARD] You called '{name}' with identical "
            f"arguments and got identical results 3+ times in a row. Stop "
            f"repeating it. Try a different tool, change the arguments "
            f"meaningfully, or report the blocker and stop."
        )

    pattern = _repeating_sequence(sigs)
    if pattern:
        desc = " → ".join(s.name for s in pattern)
        return (
            f"[SYSTEM: REPETITION GUARD] You are cycling through "
            f"[{desc}] without progress. Break the cycle: try a "
            f"different approach or stop and explain the blocker."
        )

    return None
