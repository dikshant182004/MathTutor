# ── Visual constants (unchanged) ──────────────────────────────────────────────

NODE_COLORS: dict[str, dict] = {
    "student": {
        "background": "#1a3a5c", "border": "#3b82f6",
        "highlight_background": "#1e4d80", "highlight_border": "#60a5fa",
        "hover_background": "#1e4d80",     "hover_border": "#60a5fa",
    },
    "session": {
        "background": "#1a2e1a", "border": "#22c55e",
        "highlight_background": "#1e3d1e", "highlight_border": "#4ade80",
        "hover_background": "#1e3d1e",     "hover_border": "#4ade80",
    },
    "agent": {
        "background": "#2a1f3d", "border": "#a78bfa",
        "highlight_background": "#3b2b5a", "highlight_border": "#c4b5fd",
        "hover_background": "#3b2b5a",     "hover_border": "#c4b5fd",
    },
    "tool": {
        "background": "#2d1f0a", "border": "#f59e0b",
        "highlight_background": "#3d2a0a", "highlight_border": "#fbbf24",
        "hover_background": "#3d2a0a",     "hover_border": "#fbbf24",
    },
    "episodic": {
        "background": "#1f1a2e", "border": "#818cf8",
        "highlight_background": "#2a2040", "highlight_border": "#a5b4fc",
        "hover_background": "#2a2040",     "hover_border": "#a5b4fc",
    },
    "semantic": {
        "background": "#0f2a2a", "border": "#2dd4bf",
        "highlight_background": "#0f3535", "highlight_border": "#5eead4",
        "hover_background": "#0f3535",     "hover_border": "#5eead4",
    },
    "procedural": {
        "background": "#2d1020", "border": "#f472b6",
        "highlight_background": "#3d1428", "highlight_border": "#f9a8d4",
        "hover_background": "#3d1428",     "hover_border": "#f9a8d4",
    },
}

TYPE_BADGE_STYLE: dict[str, str] = {
    "student":    "background:#0d2040;color:#60a5fa;border:1px solid #3b82f655",
    "session":    "background:#0d2010;color:#4ade80;border:1px solid #22c55e55",
    "agent":      "background:#1a1030;color:#c4b5fd;border:1px solid #a78bfa55",
    "tool":       "background:#201500;color:#fbbf24;border:1px solid #f59e0b55",
    "episodic":   "background:#110d20;color:#a5b4fc;border:1px solid #818cf855",
    "semantic":   "background:#061818;color:#5eead4;border:1px solid #2dd4bf55",
    "procedural": "background:#1a0810;color:#f9a8d4;border:1px solid #f472b655",
}

NODE_SIZES: dict[str, int] = {
    "student": 30, "session": 22, "agent": 14,
    "tool": 12, "episodic": 16, "semantic": 16, "procedural": 16,
}

NODE_SHAPES: dict[str, str] = {
    "student": "star", "session": "hexagon", "agent": "dot",
    "tool": "diamond", "episodic": "dot", "semantic": "square", "procedural": "triangleDown",
}

NODE_FONT_SIZES: dict[str, int] = {
    "student": 13, "session": 11, "agent": 9,
    "tool": 9, "episodic": 10, "semantic": 10, "procedural": 10,
}

EDGE_CONFIG: dict = {
    "color": "#1e3a5a", "highlight": "#3b82f6", "hover": "#60a5fa",
    "font_color": "#2d5070", "font_size": 9,
    "width": 1.2, "smooth_type": "curvedCW", "smooth_round": 0.15,
}

LEGEND_META: list[tuple[str, str]] = [
    ("student",    "Student root"),
    ("session",    "STM Thread"),
    ("episodic",   "Episodic LTM"),
    ("semantic",   "Semantic LTM"),
    ("procedural", "Procedural LTM"),
    ("agent",      "Agent node"),
    ("tool",       "Tool call"),
]

VIS_CDN_JS  = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"
VIS_CDN_CSS = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css"

PHYSICS_PRESETS: dict[str, dict] = {
    "Radial (default)": {
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -55,
                "centralGravity": 0.012,
                "springLength": 140,
                "springConstant": 0.06,
                "damping": 0.42,
                "avoidOverlap": 0.9,
            },
            "stabilization": {"iterations": 200, "fit": True},
        },
        "layout": {"improvedLayout": True},
    },
    "Hierarchical": {
        "physics": {"enabled": False},
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 110,
                "nodeSpacing": 100,
            }
        },
    },
    "Organic spread": {
        "physics": {
            "enabled": True,
            "solver": "repulsion",
            "repulsion": {
                "nodeDistance": 180,
                "centralGravity": 0.1,
                "springLength": 200,
                "springConstant": 0.04,
                "damping": 0.9,
            },
            "stabilization": {"iterations": 150},
        },
        "layout": {"improvedLayout": True},
    },
    "Tight cluster": {
        "physics": {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {
                "gravitationalConstant": -4000,
                "centralGravity": 0.4,
                "springLength": 80,
                "springConstant": 0.1,
                "damping": 0.5,
                "avoidOverlap": 0.5,
            },
            "stabilization": {"iterations": 180},
        },
        "layout": {"improvedLayout": True},
    },
}
