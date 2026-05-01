"""DuckDuckGo HTML web search.

Slimmed port of ml-intern's web_search_tool: hits DDG's HTML endpoint
and parses anchor tags out of the result list. No API key required.
The endpoint occasionally rate-limits or rewrites links; we surface
errors as data so the model can react rather than crash.
"""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

import httpx

from . import tool

_DDG_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = "hf-agent/0.1 (+https://github.com/anthropics/claude-code)"
_MAX_RESULTS = 8
_TIMEOUT = 15.0


class _AnchorParser(HTMLParser):
    """Pulls (title, href) pairs out of `<a class="result__a">` tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        m = {k.lower(): (v or "") for k, v in attrs}
        if "result__a" not in m.get("class", ""):
            return
        self._href = m.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            title = "".join(self._text).strip()
            if title and self._href:
                self.hits.append((title, self._href))
            self._href = None
            self._text = []


def _unwrap_ddg_redirect(url: str) -> str:
    # DDG wraps results in /l/?uddg=<encoded>. Unwrap so the model gets
    # the real destination URL.
    parsed = urlparse(url)
    if parsed.path == "/l/" and parsed.query:
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target:
            return target
    return url


@tool(
    name="web_search",
    description=(
        "Search the public web via DuckDuckGo and return up to 8 result "
        "titles + URLs. Use for general lookups, finding documentation, "
        "or discovering APIs. Does not return page contents — fetch a "
        "specific URL with `http_get` afterwards if needed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
        },
        "required": ["query"],
    },
)
async def web_search(query: str) -> str:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            resp = await client.post(
                _DDG_URL,
                data={"q": query},
                headers={"User-Agent": _USER_AGENT},
                follow_redirects=True,
            )
        except httpx.HTTPError as e:
            return f"web_search failed: {type(e).__name__}: {e}"
    if resp.status_code != 200:
        return f"web_search HTTP {resp.status_code}: {resp.text[:200]}"

    parser = _AnchorParser()
    parser.feed(resp.text)
    if not parser.hits:
        return f"No results for {query!r} (DuckDuckGo returned no result anchors)."

    lines = [f"Results for {query!r}:"]
    for i, (title, href) in enumerate(parser.hits[:_MAX_RESULTS], 1):
        lines.append(f"{i}. {title}\n   {_unwrap_ddg_redirect(href)}")
    return "\n".join(lines)
