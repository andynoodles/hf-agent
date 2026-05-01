"""Search the Hugging Face Hub for models / datasets / spaces.

Thin wrapper around `https://huggingface.co/api/{models,datasets,spaces}`.
We don't pull in `huggingface_hub` — the public list endpoints are simple
JSON GETs and keeping deps narrow makes the tool easier to audit.

Schema-as-spec: the JSON-Schema enums below mirror the values the Hub API
actually accepts (e.g. `trendingScore`, not the human word `trending`).
The model picks a value the API will accept on the first try, instead of
us silently rewriting it in the wrapper. The per-property descriptions
do the human-word → API-value translation at generation time.
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
        "Search Hugging Face Hub for models, datasets, or spaces — wraps "
        "GET /api/{models,datasets,spaces}. Returns up to 20 hits with id, "
        "downloads, likes, and tags. The schema exposes every list-endpoint "
        "param the Hub accepts: `kind` (required), `query` (free-text), "
        "`author` (org/user slug), `filter` (single canonical tag), "
        "`sort` (canonical Hub key — see enum), `direction` (-1 desc / 1 asc), "
        "`limit` (1–20). The API does NOT support time-window filters, "
        "size filters, boolean negation, regex, or fuzzy ID matching — "
        "drop those constraints from the call and surface them to the user."
    ),
    parameters={
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["models", "datasets", "spaces"],
                "description": "Which Hub catalogue to search.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Free-text keyword. Matches id, description, and tags. "
                    "Pass user terms verbatim — the API tokenises but does "
                    "not stem or fuzzy-match, so 'wino-grand' will not find "
                    "'winogrande'."
                ),
            },
            "author": {
                "type": "string",
                "description": (
                    "Restrict to a single user or org slug, e.g. 'google', "
                    "'meta-llama', 'Salesforce', 'openai-community'. Use "
                    "this when the user names a publisher."
                ),
            },
            "filter": {
                "type": "string",
                "description": (
                    "Single canonical Hub tag. One tag only — the API "
                    "rejects arrays here. Common families: task tags "
                    "('text-classification', 'question-answering', "
                    "'automatic-speech-recognition', 'text-generation'), "
                    "frameworks ('pytorch', 'tensorflow', 'jax'), language "
                    "codes ('en', 'zh', 'ru', 'ja'), licenses ('mit', "
                    "'apache-2.0', 'cc-by-4.0'). Map non-English task names "
                    "to the canonical English tag before calling."
                ),
            },
            "sort": {
                "type": "string",
                "enum": [
                    "downloads",
                    "likes",
                    "trendingScore",
                    "lastModified",
                    "createdAt",
                ],
                "description": (
                    "Canonical Hub sort key. Map human phrasing to the "
                    "API value: 'trending' / 'popular right now' → "
                    "`trendingScore` (NOT the bare word 'trending' — the "
                    "API returns HTTP 400); 'popular' / 'most used' / "
                    "'most downloaded' → `downloads`; 'most-liked' / "
                    "'best-rated' → `likes`; 'recent' / 'recently updated' "
                    "→ `lastModified`; 'newly published' / 'newest' → "
                    "`createdAt`. Default to `downloads` when the user's "
                    "ranking word is ambiguous ('good', 'best')."
                ),
            },
            "direction": {
                "type": "string",
                "enum": ["-1", "1"],
                "description": (
                    "Sort direction. `\"-1\"` for descending (most/largest/"
                    "newest first — the natural order for popularity sorts, "
                    "and the Hub default for `downloads`/`likes`/"
                    "`trendingScore`). `\"1\"` for ascending — set this only "
                    "when the user explicitly asks for least/oldest/"
                    "smallest first."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    f"Max results to return (1–{_MAX_HITS}). Honour "
                    "explicit numeric requests in the prompt ('top 3', "
                    "'5 datasets'); default to 10 otherwise."
                ),
            },
        },
        "required": ["kind"],
    },
)
async def hf_hub_search(
    kind: str,
    query: str | None = None,
    author: str | None = None,
    filter: str | None = None,
    sort: str | None = None,
    direction: str | int | None = None,
    limit: int | None = None,
) -> str:
    if kind not in {"models", "datasets", "spaces"}:
        return f"Error: kind must be one of models/datasets/spaces, got {kind!r}"

    params: dict[str, str] = {"limit": str(min(max(limit or 10, 1), _MAX_HITS))}
    if query:
        params["search"] = query
    if author:
        params["author"] = author
    if filter:
        params["filter"] = filter
    if sort:
        params["sort"] = sort
    if direction is not None and str(direction) in ("-1", "1"):
        params["direction"] = str(direction)

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
        return (
            f"No {kind} matched (query={query!r}, author={author!r}, "
            f"filter={filter!r})."
        )

    header = (
        f"Top {len(items)} {kind} (query={query!r}, author={author!r}, "
        f"filter={filter!r}, sort={sort!r}, direction={direction!r}):"
    )
    lines = [header]
    for it in items:
        ident = it.get("id") or it.get("modelId") or "?"
        downloads = it.get("downloads", "?")
        likes = it.get("likes", "?")
        tags = ", ".join((it.get("tags") or [])[:6])
        lines.append(f"- {ident}  [↓{downloads} ♥{likes}]  {tags}")
    return "\n".join(lines)
