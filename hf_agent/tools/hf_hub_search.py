"""Search the Hugging Face Hub for models / datasets / spaces.

Thin wrapper around `https://huggingface.co/api/{models,datasets,spaces}`.
We don't pull in `huggingface_hub` — the public list endpoints are simple
JSON GETs and keeping deps narrow makes the tool easier to audit.
"""
from __future__ import annotations

import json
import os

import httpx

from . import tool

_BASE = "https://huggingface.co/api"
_TIMEOUT = 15.0
_MAX_HITS = 20


@tool(
    name="hf_hub_search",
    description=(
        "Search Hugging Face Hub for models, datasets, or spaces. Returns "
        "up to 20 hits with id, downloads, likes, and tags. Use `kind` to "
        "pick which catalogue to search; `query` is a free-text match; "
        "`filter` is a single tag (e.g. 'text-classification', 'pytorch'); "
        "`sort` controls ordering (downloads | likes | trending | lastModified)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["models", "datasets", "spaces"],
                "description": "Which catalogue to search.",
            },
            "query": {"type": "string", "description": "Free-text search query."},
            "filter": {
                "type": "string",
                "description": "Single tag filter (e.g. 'text-classification').",
            },
            "sort": {
                "type": "string",
                "enum": ["downloads", "likes", "trending", "lastModified"],
                "description": "Result ordering.",
            },
            "limit": {
                "type": "integer",
                "description": f"Max results (1–{_MAX_HITS}).",
            },
        },
        "required": ["kind"],
    },
)
async def hf_hub_search(
    kind: str,
    query: str | None = None,
    filter: str | None = None,
    sort: str | None = None,
    limit: int | None = None,
) -> str:
    if kind not in {"models", "datasets", "spaces"}:
        return f"Error: kind must be one of models/datasets/spaces, got {kind!r}"

    params: dict[str, str] = {"limit": str(min(max(limit or 10, 1), _MAX_HITS))}
    if query:
        params["search"] = query
    if filter:
        params["filter"] = filter
    if sort:
        params["sort"] = sort

    headers = {"Accept": "application/json", "User-Agent": "hf-agent/0.1"}
    token = os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.get(f"{_BASE}/{kind}", params=params, headers=headers)
        except httpx.HTTPError as e:
            return f"hf_hub_search failed: {type(e).__name__}: {e}"

    if resp.status_code != 200:
        return f"hf_hub_search HTTP {resp.status_code}: {resp.text[:300]}"

    try:
        items = resp.json()
    except json.JSONDecodeError:
        return f"hf_hub_search: non-JSON response: {resp.text[:300]}"

    if not items:
        return f"No {kind} matched (query={query!r}, filter={filter!r})."

    lines = [f"Top {len(items)} {kind} (query={query!r}, filter={filter!r}, sort={sort!r}):"]
    for it in items:
        ident = it.get("id") or it.get("modelId") or "?"
        downloads = it.get("downloads", "?")
        likes = it.get("likes", "?")
        tags = ", ".join((it.get("tags") or [])[:6])
        lines.append(f"- {ident}  [↓{downloads} ♥{likes}]  {tags}")
    return "\n".join(lines)
