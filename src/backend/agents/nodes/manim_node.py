async def _call_manim_mcp(scene_code: str, scene_class: str) -> Optional[str]:
    """
    Calls the FastMCP Manim server via streamable-http transport.
    Server must be running:  python manim_mcp_server.py
    Env vars:
        MANIM_MCP_SERVER_URL  — default http://localhost:8765/mcp
    Returns absolute path to rendered video, or None on failure.

    FastMCP client.call_tool() returns a CallToolResult with:
      .data    — hydrated Python dict (preferred, populated when server returns a dict)
      .content — list of TextContent / ImageContent blocks (fallback)
      .is_error — bool
    """
    try:
        from fastmcp import Client as FastMCPClient
        import json as _json

        server_url = _get_secret("MANIM_MCP_SERVER_URL", "http://localhost:8765/mcp")
        logger.info(f"[Manim MCP] Connecting to {server_url}")

        async with FastMCPClient(server_url) as client:
            result = await client.call_tool(
                "render_manim_scene",
                {
                    "scene_code":  scene_code,
                    "scene_class": scene_class,
                    "quality":     "medium_quality",
                    "fmt":         "mp4",
                },
            )

            if getattr(result, "is_error", False):
                err = getattr(result, "data", None) or {}
                logger.error(f"[Manim MCP] Tool returned error: {err.get('error', 'unknown')}")
                return None

            # Path 1: .data is already a hydrated dict (FastMCP auto-deserialises dicts)
            payload = getattr(result, "data", None)
            if isinstance(payload, dict):
                video_path = payload.get("video_path") or payload.get("output_path")
                if video_path:
                    logger.info(f"[Manim MCP] Rendered (via .data) → {video_path}")
                    return str(video_path)

            # Path 2: .content is a list of TextContent blocks — parse JSON from .text
            for item in (getattr(result, "content", None) or []):
                text = getattr(item, "text", None)
                if not text:
                    continue
                try:
                    payload = _json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    video_path = payload.get("video_path") or payload.get("output_path")
                    if video_path:
                        logger.info(f"[Manim MCP] Rendered (via .content) → {video_path}")
                        return str(video_path)

        logger.warning("[Manim MCP] No video_path found in response")
        return None

    except ImportError:
        logger.warning("[Manim MCP] `fastmcp` not installed — pip install fastmcp")
        return None
    except Exception as exc:
        logger.error(f"[Manim MCP] Error: {exc}")
        return None

def _extract_scene_class(code: str) -> str:
    """Extract the first Scene subclass name from generated Manim code."""
    import re
    match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", code)
    return match.group(1) if match else "MathScene"


def manim_node(state: AgentState) -> AgentState:
    """
    Reads manim_scene_code from explainer_output.
    Calls the Manim MCP server to render it.
    Stores the output path in state['manim_video_path'].
    Gracefully skips if no code was generated or MCP is unavailable.
    """
    explainer = state.get("explainer_output") or {}
    scene_code = explainer.get("manim_scene_code")

    if not scene_code:
        logger.info("[Manim] No scene code — skipping.")
        return state

    scene_class = _extract_scene_class(scene_code)
    logger.info(f"[Manim] Rendering scene class: {scene_class}")

    # asyncio.run() raises RuntimeError if an event loop is already running
    # (which Streamlit does). nest_asyncio patches the loop to allow nesting.
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        logger.warning("[Manim] nest_asyncio not installed — pip install nest_asyncio")

    try:
        loop = asyncio.get_event_loop()
        video_path = loop.run_until_complete(_call_manim_mcp(scene_code, scene_class))
    except RuntimeError:
        # Last resort: new loop
        loop = asyncio.new_event_loop()
        try:
            video_path = loop.run_until_complete(_call_manim_mcp(scene_code, scene_class))
        finally:
            loop.close()

    state["manim_video_path"] = video_path

    if video_path:
        logger.info(f"[Manim] Video saved at: {video_path}")
    else:
        logger.warning("[Manim] Render failed or MCP unavailable — continuing without visualization.")

    return state
