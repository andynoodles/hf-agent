"""Generic HTTP GET against any public URL.

Covers the assignment's "any public API or database" surface area: the
agent can hit GitHub, Wikipedia, Wikidata SPARQL, Data Commons, etc.
without bespoke per-API tools. The response body is truncated so a
single huge JSON payload doesn't blow out the context window.
"""
from __future__ import annotations

import json as jsonlib

import httpx

from . import tool

_MAX_RESPONSE_CHARS = 6000
_TIMEOUT = 20.0


@tool(
    name="http_get",
    description=(
        "Issue an HTTP GET against a public URL and return status + body. "
        "Use for querying public REST APIs (GitHub, Wikipedia, Wikidata "
        "SPARQL, HF Hub, Data Commons, etc.). Optionally pass `params` "
        "for query-string args and `headers` for things like Accept. "
        "JSON responses are pretty-printed; large bodies are truncated."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL to GET."},
            "params": {
                "type": "object",
                "description": "Optional query-string parameters as a flat object of string key/value pairs.",
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP request headers as a flat object of string key/value pairs.",
            },
        },
        "required": ["url"],
    },
)
async def http_get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
) -> str:
    req_headers = {"User-Agent": "hf-agent/0.1", "Accept": "application/json"}
    if headers:
        req_headers.update(headers)

    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
        try:
            resp = await client.get(url, params=params or None, headers=req_headers)
        except httpx.HTTPError as e:
            return f"GET {url} failed: {type(e).__name__}: {e}"

    body = resp.text
    # Pretty-print JSON when possible — easier for the model to read.
    ctype = resp.headers.get("content-type", "")
    if "json" in ctype.lower():
        try:
            body = jsonlib.dumps(jsonlib.loads(body), indent=2)
        except (ValueError, TypeError):
            pass

    if len(body) > _MAX_RESPONSE_CHARS:
        body = (
            body[:_MAX_RESPONSE_CHARS]
            + f"\n… [truncated, {len(body) - _MAX_RESPONSE_CHARS} more chars]"
        )

    return f"GET {resp.url} → {resp.status_code}\n{body}"
