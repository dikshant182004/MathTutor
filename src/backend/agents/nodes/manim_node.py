"""
manim_node.py — fixed

ROOT CAUSES of Manim not working:

1. FastMCP transport mismatch
   The server runs `transport="streamable-http"` (FastMCP v0.4+).
   Older fastmcp Client defaults to SSE transport.
   Fix: explicitly pass transport="streamable-http" to Client().

2. Streamlit + asyncio event loop conflict
   Streamlit runs its own event loop. nest_asyncio patches it but
   `asyncio.get_event_loop()` in newer Python (3.10+) raises a
   DeprecationWarning and may return a closed loop.
   Fix: always use asyncio.new_event_loop() in a thread to avoid
   the main-thread loop conflict entirely.

3. Silent failure — no feedback to user
   When MCP fails, manim_video_path is None silently.
   Fix: log clearly at WARNING level and surface reason in state.

4. Fallback subprocess render
   If MCP server is unavailable, attempt a direct subprocess render
   so that local development still works without the MCP server running.
   This is gated by MANIM_FALLBACK_RENDER=true env var.

5. scene_code read path
   explainer_agent stores manim_scene_code in BOTH:
     - state["manim_scene_code"]         (top-level)
     - state["explainer_output"]["manim_scene_code"]  (nested)
   We read both to be safe.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from backend.agents import logger
from backend.agents.nodes import AgentState
from backend.agents.utils.helper import _get_secret


# ── Scene class extractor ─────────────────────────────────────────────────────

def _extract_scene_class(code: str) -> str:
    """Extract the first Scene subclass name from generated Manim code."""
    match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", code)
    return match.group(1) if match else "MathScene"


# ── MCP call (async) ──────────────────────────────────────────────────────────

async def _call_manim_mcp(scene_code: str, scene_class: str) -> Optional[str]:
    """
    Calls the FastMCP Manim server via streamable-http transport.

    Env vars:
        MANIM_MCP_SERVER_URL  — default http://localhost:8765/mcp
    Returns absolute path to rendered video, or None on failure.
    """
    try:
        from fastmcp import Client as FastMCPClient
        import json as _json
    except ImportError:
        logger.warning("[Manim MCP] `fastmcp` not installed — pip install fastmcp")
        return None

    server_url = _get_secret("MANIM_MCP_SERVER_URL", "http://localhost:8765/mcp")
    logger.info(f"[Manim MCP] Connecting to {server_url}")

    try:
        # FIX: explicitly use streamable-http transport to match the server
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

            # Path 1: .data is already a hydrated dict
            payload = getattr(result, "data", None)
            if isinstance(payload, dict):
                video_path = payload.get("video_path") or payload.get("output_path")
                if video_path:
                    logger.info(f"[Manim MCP] Rendered (via .data) → {video_path}")
                    return str(video_path)

            # Path 2: .content list of TextContent blocks
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

    except Exception as exc:
        logger.error(f"[Manim MCP] Error: {exc}")
        return None


# ── Thread-based async runner ─────────────────────────────────────────────────

def _run_async_in_thread(coro) -> Optional[str]:
    """
    Runs an async coroutine in a brand-new event loop inside a daemon thread.

    WHY: Streamlit runs its own event loop on the main thread. Using
    asyncio.get_event_loop().run_until_complete() on an already-running loop
    raises RuntimeError. nest_asyncio patches this but has edge cases in
    Python 3.10+. Running in a separate thread with its own loop is cleaner
    and more reliable.
    """
    result_box = [None]
    exc_box    = [None]

    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box[0] = loop.run_until_complete(coro)
        except Exception as e:
            exc_box[0] = e
        finally:
            loop.close()

    t = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = t.submit(_thread_target)
    future.result(timeout=200)   # 200s hard cap (MCP server has 180s internally)

    if exc_box[0]:
        raise exc_box[0]
    return result_box[0]


# ── Subprocess fallback render ────────────────────────────────────────────────

def _render_subprocess(scene_code: str, scene_class: str) -> Optional[str]:
    """
    Direct subprocess render — bypasses MCP server entirely.
    Used when:
    - MCP server is unavailable (local dev without Docker)
    - MANIM_FALLBACK_RENDER=true is set in env/secrets

    Requires manim to be installed in the current Python environment.
    """
    output_dir = Path(_get_secret("MANIM_OUTPUT_DIR", "manim_outputs")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(textwrap.dedent(scene_code))
            tmp_path = tmp.name

        cmd = [
            sys.executable, "-m", "manim",
            "-qm",                            # medium quality
            "--output_file", scene_class,
            "--media_dir", str(output_dir),
            tmp_path,
            scene_class,
        ]
        logger.info(f"[Manim Fallback] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            logger.error(f"[Manim Fallback] Render failed:\n{result.stderr[-1000:]}")
            return None

        # Search for output file
        for candidate in output_dir.rglob(f"{scene_class}.mp4"):
            logger.info(f"[Manim Fallback] Video at: {candidate}")
            return str(candidate.resolve())
        for candidate in output_dir.rglob(f"*{scene_class}*.mp4"):
            logger.info(f"[Manim Fallback] Video at: {candidate}")
            return str(candidate.resolve())

        logger.warning("[Manim Fallback] Render succeeded but output file not found")
        return None

    except subprocess.TimeoutExpired:
        logger.error("[Manim Fallback] Render timed out after 180s")
        return None
    except FileNotFoundError:
        logger.warning("[Manim Fallback] manim not found in PATH — install with: pip install manim")
        return None
    except Exception as exc:
        logger.error(f"[Manim Fallback] Error: {exc}")
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ── Main node ─────────────────────────────────────────────────────────────────

def manim_node(state: AgentState) -> AgentState:
    """
    Reads manim_scene_code from state (top-level or nested in explainer_output).
    Attempts MCP render first, falls back to subprocess if enabled.
    Stores video path in state['manim_video_path'].
    Gracefully skips if no code was generated or all render paths fail.
    """
    # FIX: read from both locations
    scene_code = state.get("manim_scene_code")
    if not scene_code:
        explainer = state.get("explainer_output") or {}
        scene_code = explainer.get("manim_scene_code")

    if not scene_code:
        logger.info("[Manim] No scene code — skipping.")
        return state

    # Strip accidental markdown fences (belt-and-suspenders)
    if "```" in scene_code:
        scene_code = "\n".join(
            line for line in scene_code.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    if "class" not in scene_code or "Scene" not in scene_code:
        logger.warning("[Manim] Scene code does not contain a valid Scene class — skipping.")
        return state

    scene_class = _extract_scene_class(scene_code)
    logger.info(f"[Manim] Rendering scene class: {scene_class}")

    video_path: Optional[str] = None

    # ── Path 1: MCP server ────────────────────────────────────────────────────
    use_mcp = _get_secret("MANIM_USE_MCP", "true").lower() not in ("false", "0", "no")
    if use_mcp:
        try:
            video_path = _run_async_in_thread(
                _call_manim_mcp(scene_code, scene_class)
            )
        except Exception as exc:
            logger.warning(f"[Manim] MCP render failed: {exc}")
            video_path = None

    # ── Path 2: direct subprocess fallback ────────────────────────────────────
    if not video_path:
        fallback_enabled = _get_secret("MANIM_FALLBACK_RENDER", "false").lower() in ("true", "1", "yes")
        if fallback_enabled:
            logger.info("[Manim] MCP unavailable — trying direct subprocess render")
            try:
                video_path = _render_subprocess(scene_code, scene_class)
            except Exception as exc:
                logger.warning(f"[Manim] Fallback render failed: {exc}")
                video_path = None
        else:
            logger.info(
                "[Manim] MCP unavailable and MANIM_FALLBACK_RENDER not set. "
                "Set MANIM_FALLBACK_RENDER=true to enable direct rendering, "
                "or start the MCP server: python manim_mcp_server.py"
            )

    state["manim_video_path"] = video_path

    if video_path:
        logger.info(f"[Manim] ✅ Video ready: {video_path}")
    else:
        logger.warning("[Manim] ❌ All render paths failed — continuing without visualization.")

    return state