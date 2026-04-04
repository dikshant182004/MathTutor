from __future__ import annotations

import asyncio
import concurrent.futures
import json
from backend.agents import Any, Optional

from backend.agents import logger
from backend.agents.nodes.tools.mcp import TAVILY_REMOTE_MCP_BASE, _TOOL_SEARCH
from backend.agents.utils.helper import _get_secret


def _get_server_url() -> str:
    """Build the Tavily remote MCP URL with the API key embedded."""
    api_key = _get_secret("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. "
            "Add it to your .env or Streamlit secrets to enable web search."
        )
    return f"{TAVILY_REMOTE_MCP_BASE}?tavilyApiKey={api_key}"


def _parse_mcp_result(result: Any) -> Optional[dict]:
    """
    Extract the payload dict from a FastMCP ToolResult.

    FastMCP returns results in two shapes — handle both:
      Shape 1: result.data is already a dict
      Shape 2: result.content is a list of TextContent blocks with .text = JSON string
    """
    # Shape 1
    payload = getattr(result, "data", None)
    if isinstance(payload, dict):
        return payload

    # Shape 2
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            # Sometimes it's already plain text — return as-is wrapped
            return {"raw_text": text}

    return None


def _format_search_results(payload: dict) -> str:
    """Convert Tavily search result dict into clean markdown for the LLM."""
    lines: list[str] = []

    # Tavily AI direct answer (when available)
    answer = payload.get("answer") or payload.get("direct_answer", "")
    if answer:
        lines.append(f"**Direct Answer:** {answer}\n")

    results: list[dict] = payload.get("results") or []
    if not results:
        # Fallback: maybe the whole payload IS the results list
        if isinstance(payload.get("raw_text"), str):
            return payload["raw_text"]
        return "No results returned from Tavily."

    lines.append(f"**Search Results** ({len(results)} found):\n")
    for i, r in enumerate(results, 1):
        title     = r.get("title",    "No title")
        url       = r.get("url",      "")
        content   = (r.get("content") or "").strip()
        score     = r.get("score",    0.0)
        published = r.get("published_date", "")

        date_str = f" · {published}" if published else ""
        lines.append(
            f"**{i}. {title}**{date_str}  \n"
            f"URL: {url}  \n"
            f"Relevance: {score:.2f}  \n"
            f"{content[:500]}\n"
        )

    return "\n---\n".join(lines)


# ── Async MCP calls ───────────────────────────────────────────────────────────

async def _async_tavily_search(
    query: str,
    search_depth: str  = "advanced",
    topic: str         = "general",
    max_results: int   = 5,
) -> str:
    """
    Calls the Tavily MCP `tavily-search` tool.
    """
    try:
        from fastmcp import Client as FastMCPClient
    except ImportError:
        logger.warning("[TavilyMCP] fastmcp not installed — pip install fastmcp")
        return "Web search unavailable: fastmcp not installed."

    server_url = _get_server_url()
    logger.info(f"[TavilyMCP] search: '{query}' depth={search_depth}")

    try:
        async with FastMCPClient(server_url) as client:
            result = await client.call_tool(
                _TOOL_SEARCH,
                {
                    "query":          query,
                    "search_depth":   search_depth,
                    "topic":          topic,
                    "max_results":    max_results,
                },
            )

        if getattr(result, "is_error", False):
            err = _parse_mcp_result(result) or {}
            msg = err.get("error") or err.get("message") or "unknown error"
            logger.error(f"[TavilyMCP] Tool error: {msg}")
            return ""

        payload = _parse_mcp_result(result)
        if not payload:
            logger.warning("[TavilyMCP] Empty response from tavily-search")
            return ""

        formatted = _format_search_results(payload)
        logger.info(f"[TavilyMCP] search ok — {len(formatted)} chars")
        return formatted

    except Exception as exc:
        logger.error(f"[TavilyMCP] search error: {exc}")
        return ""


# ── Thread-safe async runner  ─────────────────────────

def _run_in_thread(coro, timeout: int = 30) -> str:
    """
    Run an async coroutine in a fresh event loop inside a daemon thread.
    """
    result_box: list[Any] = [None]
    exc_box:    list[Any] = [None]

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box[0] = loop.run_until_complete(coro)
        except Exception as e:
            exc_box[0] = e
        finally:
            loop.close()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future   = executor.submit(_target)
    try:
        future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return f"Search timed out after {timeout}s."
    finally:
        executor.shutdown(wait=False)

    if exc_box[0]:
        logger.error(f"[TavilyMCP] Thread error: {exc_box[0]}")
        return f"Search error: {exc_box[0]}"

    return result_box[0] or "No results."


# ── Public sync API (called by LangChain tools) ───────────────────────────────

def tavily_mcp_search(
    query: str,
    search_depth: str   = "advanced",
    topic: str          = "general",
    max_results: int    = 5
) -> str:
    """
    Synchronous wrapper — safe to call from Streamlit / LangChain tools.

    Args:
        query         : Natural language search query.
        search_depth  : "basic" (fast) or "advanced" (thorough, uses more credits).
        topic         : "general" | "news" — use "news" for recent discoveries.
        max_results   : Number of results to return (1-10).
        

    Returns:
        Formatted markdown string with direct answer + ranked results.
    """
    return _run_in_thread(
        _async_tavily_search(
            query         = query,
            search_depth  = search_depth,
            topic         = topic,
            max_results   = max_results,
        ),
        timeout=30,
    )
