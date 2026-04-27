from __future__ import annotations

import asyncio
import os

from . import context as tool_context, tool

_MAX_OUTPUT_CHARS = 4000  # truncate per-stream so the model isn't flooded
_TIMEOUT_DEFAULT = 30
_TIMEOUT_MAX = 600  # cap per single wait window; renewed on each "keep waiting"


def _clip(label: str, text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return f"--- {label} ---\n{text}"
    head = text[:_MAX_OUTPUT_CHARS]
    return (
        f"--- {label} (truncated, {len(text)} chars total, showing first {_MAX_OUTPUT_CHARS}) ---\n"
        f"{head}"
    )


async def _drain(stream: asyncio.StreamReader | None) -> bytes:
    if stream is None:
        return b""
    return await stream.read()


@tool(
    requires_approval=True,
    name="run_shell",
    description=(
        "Execute a shell command via /bin/sh in the current working "
        "directory and return its exit code, stdout, and stderr. Use this "
        "to inspect files, list directories, run scripts, query git, etc. "
        "Output is truncated if very large. If the command is still running "
        "when the timeout fires, the user is asked whether to keep waiting; "
        "the command is only killed if the user explicitly declines."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": (
                    f"Seconds to wait before asking the user whether to keep "
                    f"waiting. Default {_TIMEOUT_DEFAULT}, capped at {_TIMEOUT_MAX}."
                ),
            },
        },
        "required": ["command"],
    },
)
async def run_shell(command: str, timeout_seconds: int | None = None) -> str:
    timeout = timeout_seconds if timeout_seconds is not None else _TIMEOUT_DEFAULT
    timeout = max(1, min(_TIMEOUT_MAX, int(timeout)))

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=os.getcwd(),
    )

    # Drain pipes in background so the process can't deadlock filling
    # OS-level pipe buffers while we're waiting on `proc.wait()`.
    stdout_task = asyncio.create_task(_drain(proc.stdout))
    stderr_task = asyncio.create_task(_drain(proc.stderr))

    ctx = tool_context.get()
    elapsed = 0
    killed_by_user = False

    while True:
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            break  # process exited on its own
        except asyncio.TimeoutError:
            elapsed += timeout
            keep_waiting = False
            if ctx is not None:
                keep_waiting = await ctx.confirm(
                    title="⏳ Command still running",
                    body=(
                        f"[b]$ {command}[/]\n\n"
                        f"Elapsed: [b]{elapsed}s[/]\n"
                        f"Keep waiting another [b]{timeout}s[/]?"
                    ),
                )
            if not keep_waiting:
                proc.kill()
                await proc.wait()
                killed_by_user = True
                break

    stdout_b = await stdout_task
    stderr_b = await stderr_task

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")

    header_lines = [f"$ {command}"]
    if killed_by_user:
        header_lines.append(f"killed by user after {elapsed}s")
    header_lines.append(f"exit_code: {proc.returncode}")

    return (
        "\n".join(header_lines)
        + "\n"
        + _clip("stdout", stdout)
        + "\n"
        + _clip("stderr", stderr)
    )
