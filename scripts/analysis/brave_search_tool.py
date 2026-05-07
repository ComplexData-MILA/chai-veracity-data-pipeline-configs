"""Brave Search API as an openai-agents FunctionTool.

Exposes a factory that returns a tool the LLM can call to search the web via
the Brave Search API, optionally with a freshness date-range filter.

Usage:
    from scripts.analysis.brave_search_tool import create_brave_search_tool

    # Full search (no date filter)
    tool = create_brave_search_tool()

    # Time-limited search (2020-01-01 through two weeks ago)
    tool = create_brave_search_tool(
        freshness_start="2020-01-01",
        freshness_end="2026-04-23",
    )
"""

from __future__ import annotations

import logging
import os
import re as _re
from datetime import date, timedelta

import httpx
from agents import function_tool

logger = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
MAX_RESULTS = 10


def create_brave_search_tool(
    freshness_start: str | None = None,
    freshness_end: str | None = None,
):
    """Return a Brave Search tool with optional freshness date-range filtering.

    Parameters
    ----------
    freshness_start : str or None
        Start date in YYYY-MM-DD format (inclusive).
    freshness_end : str or None
        End date in YYYY-MM-DD format (inclusive).

    If both are provided, the tool adds ``freshness=<start>to<end>`` to every
    search request, restricting results to that date range.

    Returns
    -------
    FunctionTool
    """
    if bool(freshness_start) != bool(freshness_end):
        raise ValueError(
            "Both freshness_start and freshness_end must be provided, or neither."
        )

    freshness_value = ""
    if freshness_start and freshness_end:
        freshness_value = f"{freshness_start}to{freshness_end}"

    # Capture freshness_value in a closure so each tool instance is independent.

    @function_tool(
        name_override="brave_web_search",
        description_override=(
            "Search the web using the Brave Search API. "
            "Returns up to 10 results with title, URL, age, and description. "
            "Use this to find reliable sources to fact-check a claim."
        ),
    )
    async def brave_web_search(query: str) -> str:
        """Search the web via Brave Search API.

        Parameters
        ----------
        query : str
            The search query string.
        """
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            return "Error: BRAVE_SEARCH_API_KEY environment variable not set."

        params: dict[str, str | int] = {
            "q": query,
            "count": MAX_RESULTS,
            "search_lang": "en",
        }
        if freshness_value:
            params["freshness"] = freshness_value

        headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            try:
                resp = await client.get(BRAVE_API_URL, params=params, headers=headers)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning("Brave Search HTTP %d", exc.response.status_code)
                return f"Brave Search API error (HTTP {exc.response.status_code})."
            except httpx.RequestError as exc:
                logger.warning("Brave Search request failed: %s", exc)
                return f"Brave Search request error: {exc}"

        data = resp.json()
        results = (data.get("web") or {}).get("results", [])
        if not results:
            return f"No web results found for query: {query}"

        lines = [f"Brave Web Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            desc = _re.sub(r"<[^>]+>", "", r.get("description", ""))
            age = r.get("age", "")
            lines.append(
                f"{i}. {title}\n"
                f"   URL: {url}\n"
                f"   Age: {age}\n"
                f"   Description: {desc}\n"
            )

        return "\n".join(lines)

    return brave_web_search


def get_stale_freshness_range() -> tuple[str, str]:
    """Return (start, end) for the staleness filter.

    Start: 2020-01-01 (fixed).
    End: two weeks before today.
    """
    end = date.today() - timedelta(days=14)
    return "2020-01-01", end.isoformat()
