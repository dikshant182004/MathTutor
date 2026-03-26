from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP(
    name="ManimRenderer",
    instructions=(
        "Renders Manim Community Edition Python scenes to mp4 video. "
        "Pass complete, self-contained scene code and the class name to render."
    ),
)

OUTPUT_DIR = Path(os.getenv("MANIM_OUTPUT_DIR", "./manim_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXECUTABLE = os.getenv("MANIM_PYTHON") or sys.executable
print(f"[ManimMCP] Using Python for renders: {PYTHON_EXECUTABLE}", flush=True)
print(f"[ManimMCP] Output directory: {OUTPUT_DIR.resolve()}", flush=True)


@mcp.tool
def render_manim_scene(
    scene_code:  str,
    scene_class: str,
    quality:     str = "medium_quality",
    fmt:         str = "mp4",
) -> dict:
    """
    Render a Manim Community Edition scene to video.

    Parameters
    ----------
    scene_code  : Complete Python source containing the scene class (from manim import *).
    scene_class : Name of the Scene subclass to render, e.g. 'DerivativeScene'.
    quality     : 'low_quality' | 'medium_quality' | 'high_quality'  (default: medium_quality)
    fmt         : Output format — 'mp4' or 'gif'  (default: mp4)

    Returns
    -------
    {
        "success":    bool,
        "video_path": str | None,
        "error":      str | None,
        "scene_class": str,
        "quality":     str,
    }
    """
    quality_flags = {
        "low_quality":    "-ql",
        "medium_quality": "-qm",
        "high_quality":   "-qh",
    }
    q_flag = quality_flags.get(quality, "-qm")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as tmp:
            tmp.write(textwrap.dedent(scene_code))
            tmp_path = tmp.name

        print(f"[ManimMCP] Rendering {scene_class} (quality={quality})", flush=True)

        cmd = [
            PYTHON_EXECUTABLE, "-m", "manim",
            q_flag,
            "--output_file", scene_class,
            "--media_dir", str(OUTPUT_DIR),
            tmp_path,
            scene_class,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            print(f"[ManimMCP] Render failed: {result.stderr[-500:]}", flush=True)
            return {
                "success":     False,
                "video_path":  None,
                "error":       result.stderr[-2000:],
                "scene_class": scene_class,
                "quality":     quality,
            }

        video_path = _find_output(scene_class, fmt)

        if not video_path:
            return {
                "success":     False,
                "video_path":  None,
                "error":       f"Render succeeded but output file not found for '{scene_class}'",
                "scene_class": scene_class,
                "quality":     quality,
            }

        print(f"[ManimMCP] ✅ Video ready: {video_path}", flush=True)
        return {
            "success":     True,
            "video_path":  str(video_path),
            "error":       None,
            "scene_class": scene_class,
            "quality":     quality,
        }

    except subprocess.TimeoutExpired:
        return {
            "success":     False,
            "video_path":  None,
            "error":       "Render timed out after 180 seconds",
            "scene_class": scene_class,
            "quality":     quality,
        }
    except Exception as exc:
        return {
            "success":     False,
            "video_path":  None,
            "error":       str(exc),
            "scene_class": scene_class,
            "quality":     quality,
        }
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@mcp.resource("renders://list")
def list_renders() -> dict:
    """List all previously rendered video files in the output directory."""
    videos = sorted(OUTPUT_DIR.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    gifs   = sorted(OUTPUT_DIR.rglob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "mp4_files":  [str(v) for v in videos[:20]],
        "gif_files":  [str(g) for g in gifs[:20]],
        "output_dir": str(OUTPUT_DIR),
    }


def _find_output(scene_class: str, fmt: str) -> Path | None:
    """
    Search OUTPUT_DIR recursively for a rendered file matching the scene class.
    """
    ext = f".{fmt}"

    for candidate in OUTPUT_DIR.rglob(f"{scene_class}{ext}"):
        return candidate.resolve()

    for candidate in OUTPUT_DIR.rglob(f"*{scene_class}*{ext}"):
        return candidate.resolve()

    now = time.time()
    candidates = [
        p for p in OUTPUT_DIR.rglob(f"*{ext}")
        if now - p.stat().st_mtime < 300
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime).resolve()

    return None


if __name__ == "__main__":
    port = int(os.getenv("MANIM_SERVER_PORT", "8765"))
    print(f"[ManimMCP] Starting server on http://0.0.0.0:{port}/mcp", flush=True)
    print(f"[ManimMCP] Health check: http://localhost:{port}/mcp (GET)", flush=True)
    # FIX: explicitly specify transport, host, and port
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port,
    )