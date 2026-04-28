"""Read-only access to the Hugging Face Datasets Server API.

Mirrors the surface in the `huggingface-datasets` skill:
is-valid / splits / first-rows / rows / search / filter / size / statistics.
The agent picks an `endpoint` and we forward query params verbatim.
This is intentionally a thin proxy — the assignment is about turning NL
into the right structured params, not about us hiding the API shape.
"""
from __future__ import annotations

import json
import os

import httpx

from . import tool

_BASE = "https://datasets-server.huggingface.co"
_TIMEOUT = 20.0
_MAX_RESPONSE_CHARS = 6000

_ENDPOINTS = (
    "is-valid",
    "splits",
    "first-rows",
    "rows",
    "search",
    "filter",
    "parquet",
    "size",
    "statistics",
)


@tool(
    name="hf_dataset_viewer",
    description=(
        "Query the Hugging Face Datasets Server (https://datasets-server."
        "huggingface.co). Pick an endpoint and pass `dataset` plus any "
        "endpoint-specific params (config, split, offset, length, query, "
        "where). Common flow: is-valid → splits → first-rows → rows. "
        "`length` for row endpoints maxes at 100; `offset` is 0-based."
    ),
    parameters={
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "enum": list(_ENDPOINTS),
                "description": "Dataset Viewer endpoint to call.",
            },
            "dataset": {
                "type": "string",
                "description": "Dataset id, e.g. 'glue' or 'squad'.",
            },
            "config": {"type": "string", "description": "Dataset config name."},
            "split": {"type": "string", "description": "Split name (train/validation/test)."},
            "offset": {"type": "integer", "description": "0-based row offset."},
            "length": {"type": "integer", "description": "Row count (max 100)."},
            "query": {"type": "string", "description": "Search text (for /search)."},
            "where": {"type": "string", "description": "Filter predicate (for /filter)."},
        },
        "required": ["endpoint", "dataset"],
    },
)
async def hf_dataset_viewer(
    endpoint: str,
    dataset: str,
    config: str | None = None,
    split: str | None = None,
    offset: int | None = None,
    length: int | None = None,
    query: str | None = None,
    where: str | None = None,
) -> str:
    if endpoint not in _ENDPOINTS:
        return f"Error: endpoint must be one of {_ENDPOINTS}, got {endpoint!r}"

    params: dict[str, str] = {"dataset": dataset}
    for k, v in {
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
        "query": query,
        "where": where,
    }.items():
        if v is not None:
            params[k] = str(v)

    headers = {"Accept": "application/json", "User-Agent": "hf-agent/0.1"}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.get(f"{_BASE}/{endpoint}", params=params, headers=headers)
        except httpx.HTTPError as e:
            return f"hf_dataset_viewer failed: {type(e).__name__}: {e}"

    body = resp.text
    try:
        body = json.dumps(json.loads(body), indent=2)
    except (ValueError, TypeError):
        pass
    if len(body) > _MAX_RESPONSE_CHARS:
        body = body[:_MAX_RESPONSE_CHARS] + f"\n… [truncated, {len(body) - _MAX_RESPONSE_CHARS} more chars]"

    return f"GET /{endpoint} {dict(params)} → {resp.status_code}\n{body}"
