"""
memory_viz.py  —  Neo4j-style memory graph visualiser
Location: frontend/pages/memory_viz.py

Streamlit REQUIRES multi-page files to live in pages/ — so this file stays here.
All graph assets (HTML/CSS/JS) live in frontend/graph/.
memory_graph_reader lives in backend/agents/utils/.

Directory layout expected:
    frontend/
    ├── app.py
    ├── pages/
    │   ├── memory_viz.py          ← this file
    │   ├── graph.html
    │   ├── graph.css
    │   └── graph.js
    └── templates/
        └── styles.css

    backend/agents/utils/
    └── memory_graph_reader.py

Navigation:
  • Back to tutor : st.switch_page("app.py")   (relative to frontend/)
  • Auth gate     : redirects to app.py if not logged in
"""
from __future__ import annotations
import json
from pathlib import Path

import streamlit as st

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="Memory Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.user.is_logged_in:
    st.switch_page("app.py")

# ── Path helpers ──────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent        # frontend/pages/
_FRONTEND  = _HERE.parent                 # frontend/
_GRAPH_DIR = _HERE                        # frontend/pages/   ← graph.html/css/js live here too
_TEMPLATES = _FRONTEND / "templates"      # frontend/templates/

# ── Global CSS ────────────────────────────────────────────────────────────────
try:
    _app_css = (_TEMPLATES / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{_app_css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Extra page-level overrides
st.markdown("""
<style>
section.main > div { padding-top: 0.5rem !important; }
[data-testid="stSidebar"] { background: #0c1020 !important; }
[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Backend imports ───────────────────────────────────────────────────────────
# memory_graph_reader is in backend/agents/utils/ (not frontend)
try:
    import redis
    from backend.agents.graph import checkpointer
    from backend.agents.utils.db_utils import get_thread_history
    from backend.agents.utils.memory_graph_reader import build_graph_data
    _BACKEND_OK = True
except Exception as _import_err:
    _BACKEND_OK = False
    _import_err_msg = str(_import_err)

# ── Redis client ──────────────────────────────────────────────────────────────
@st.cache_resource
def _get_redis():
    try:
        import os
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        r    = redis.Redis(host=host, port=port, db=0)
        r.ping()
        return r
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  VISUAL CONSTANTS  (neo4j palette)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS PRESETS
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_legend_rows(visible_types: set[str]) -> str:
    rows = []
    for node_type, label in LEGEND_META:
        if node_type not in visible_types:
            continue
        color = NODE_COLORS[node_type]["border"]
        rows.append(
            f'<div class="row">'
            f'<div class="dot" style="background:{color};box-shadow:0 0 5px {color}88"></div>'
            f'<span>{label}</span>'
            f'</div>'
        )
    return "\n".join(rows)


def render_graph_html(
    graph_data: dict,
    layout_preset: str,
    show_labels: bool,
    show_edge_labels: bool,
) -> str:
    """
    Reads graph.html / graph.css / graph.js from disk, injects all tokens,
    returns the complete HTML string for st.components.v1.html().
    """
    try:
        html_tpl = (_GRAPH_DIR / "graph.html").read_text(encoding="utf-8")
        css_src  = (_GRAPH_DIR / "graph.css").read_text(encoding="utf-8")
        js_src   = (_GRAPH_DIR / "graph.js").read_text(encoding="utf-8")
    except FileNotFoundError as e:
        return f"<pre style='color:red'>Missing file: {e}</pre>"

    layout_opts = PHYSICS_PRESETS.get(layout_preset, PHYSICS_PRESETS["Radial (default)"])
    physics_on  = layout_opts["physics"].get("enabled", True)

    # Filter edge labels if disabled
    filtered_edges = []
    for e in graph_data.get("edges", []):
        fe = dict(e)
        if not show_edge_labels:
            fe["label"] = ""
        filtered_edges.append(fe)

    graph_json = json.dumps({
        "nodes": graph_data.get("nodes", []),
        "edges": filtered_edges,
    })

    visible_types = {n["type"] for n in graph_data.get("nodes", [])}

    replacements = {
        "%%VIS_CDN_JS%%":       VIS_CDN_JS,
        "%%VIS_CDN_CSS%%":      VIS_CDN_CSS,
        "%%INLINE_CSS%%":       css_src,
        "%%INLINE_JS%%":        js_src,
        "%%GRAPH_JSON%%":       graph_json,
        "%%LAYOUT_OPTIONS%%":   json.dumps(layout_opts),
        "%%PHYSICS_ENABLED%%":  str(physics_on).lower(),
        "%%PHYSICS_ACTIVE%%":   "active" if physics_on else "",
        "%%PHYSICS_LABEL%%":    "⚡ Physics" if physics_on else "❄ Frozen",
        "%%NODE_COLORS%%":      json.dumps(NODE_COLORS),
        "%%TYPE_BADGE_STYLE%%": json.dumps(TYPE_BADGE_STYLE),
        "%%NODE_SIZES%%":       json.dumps(NODE_SIZES),
        "%%NODE_SHAPES%%":      json.dumps(NODE_SHAPES),
        "%%NODE_FONT_SIZES%%":  json.dumps(NODE_FONT_SIZES),
        "%%EDGE_CONFIG%%":      json.dumps(EDGE_CONFIG),
        "%%LEGEND_ROWS%%":      _build_legend_rows(visible_types),
        "%%SHOW_LABELS%%":      str(show_labels).lower(),
    }

    result = html_tpl
    for token, value in replacements.items():
        result = result.replace(token, value)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def _load_graph(student_id: str, include_agents: bool, max_threads: int) -> dict:
    if not _BACKEND_OK:
        return {"nodes": [], "edges": []}
    rc = _get_redis()
    if rc is None:
        return {"nodes": [], "edges": []}
    return build_graph_data(
        student_id          = student_id,
        redis_client        = rc,
        checkpointer        = checkpointer,
        get_thread_history  = get_thread_history,
        max_threads         = max_threads,
        include_agent_nodes = include_agents,
    )


def _filter_graph(
    raw: dict,
    visible_types: list[str],
) -> dict:
    """Keep only nodes whose type is in visible_types; prune dangling edges."""
    type_set    = set(visible_types)
    nodes       = [n for n in raw.get("nodes", []) if n.get("type") in type_set]
    valid_ids   = {n["id"] for n in nodes}
    edges       = [
        e for e in raw.get("edges", [])
        if e.get("from") in valid_ids and e.get("to") in valid_ids
    ]
    return {"nodes": nodes, "edges": edges}


# ══════════════════════════════════════════════════════════════════════════════
#  TOP NAV BAR
# ══════════════════════════════════════════════════════════════════════════════

nav_left, nav_right = st.columns([8, 2], vertical_alignment="center")
with nav_left:
    st.markdown(
        "<h2 style='margin:0;color:#c8d8f0;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:0.04em;'>🧠 Memory Graph</h2>",
        unsafe_allow_html=True,
    )
with nav_right:
    if st.button("🧮 Tutor", use_container_width=True, help="Back to Math Tutor"):
        st.switch_page("app.py")   # relative to frontend/

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  — filters & controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
        "color:#3b82f6;letter-spacing:0.1em;text-transform:uppercase;"
        "padding-bottom:6px;border-bottom:1px solid #1e2d45;margin-bottom:14px'>"
        "⬡ Graph Controls</div>",
        unsafe_allow_html=True,
    )

    # ── Layout preset ─────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#2d5070;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:4px'>Layout</div>",
        unsafe_allow_html=True,
    )
    layout_choice = st.radio(
        "layout",
        list(PHYSICS_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Node type visibility ──────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#2d5070;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:6px'>Visible Node Types</div>",
        unsafe_allow_html=True,
    )

    all_node_types = [t for t, _ in LEGEND_META]
    type_visibility: dict[str, bool] = {}

    for node_type, label in LEGEND_META:
        color = NODE_COLORS[node_type]["border"]
        col_dot, col_chk = st.columns([1, 5])
        with col_dot:
            st.markdown(
                f"<div style='width:10px;height:10px;border-radius:50%;"
                f"background:{color};box-shadow:0 0 5px {color}88;"
                f"margin-top:8px'></div>",
                unsafe_allow_html=True,
            )
        with col_chk:
            type_visibility[node_type] = st.checkbox(
                label,
                value=True,
                key=f"vis_{node_type}",
            )

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Display options ───────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#2d5070;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:6px'>Display</div>",
        unsafe_allow_html=True,
    )
    show_labels      = st.toggle("Node labels",      value=True)
    show_edge_labels = st.toggle("Edge labels",       value=False)
    include_agents   = st.toggle("Agent/Tool nodes",  value=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Thread depth ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;color:#2d5070;text-transform:uppercase;"
        "letter-spacing:0.08em;margin-bottom:4px'>Max Threads</div>",
        unsafe_allow_html=True,
    )
    max_threads = st.slider("max_threads", 1, 30, 15, label_visibility="collapsed")

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Refresh ───────────────────────────────────────────────────────────────
    if st.button("🔄 Refresh Graph", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── Graph stats (populated after load) ───────────────────────────────────
    stats_ph = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — load + render
# ══════════════════════════════════════════════════════════════════════════════

student_id = st.session_state.get("student_id")

if not student_id:
    st.warning("No active session found. Please return to the tutor and log in.")
    st.stop()

if not _BACKEND_OK:
    st.error(f"Backend unavailable: {_import_err_msg}")
    st.stop()

with st.spinner("Loading memory graph…"):
    raw_graph = _load_graph(student_id, include_agents, max_threads)

# Apply visibility filters
visible_types = [t for t, checked in type_visibility.items() if checked]
graph_data    = _filter_graph(raw_graph, visible_types)

n_nodes = len(graph_data["nodes"])
n_edges = len(graph_data["edges"])

# ── Sidebar stats update ──────────────────────────────────────────────────────
with stats_ph:
    st.markdown(
        f"<div style='background:#0d1424;border:1px solid #1e2d45;border-radius:8px;"
        f"padding:10px 12px;font-family:JetBrains Mono,monospace;font-size:0.72rem;"
        f"color:#4a6080;margin-top:4px'>"
        f"<div style='color:#3b82f6;margin-bottom:5px;font-size:0.65rem;"
        f"text-transform:uppercase;letter-spacing:0.1em'>Graph Stats</div>"
        f"<div>Nodes &nbsp;<span style='color:#7eb8f7'>{n_nodes}</span></div>"
        f"<div>Edges &nbsp;<span style='color:#7eb8f7'>{n_edges}</span></div>"
        f"<div>Threads <span style='color:#7eb8f7'>"
        f"{sum(1 for n in graph_data['nodes'] if n.get('type')=='session')}"
        f"</span></div>"
        f"<div>LTM &nbsp;&nbsp;&nbsp;"
        f"<span style='color:#7eb8f7'>"
        f"{sum(1 for n in graph_data['nodes'] if n.get('type') in ('episodic','semantic','procedural'))}"
        f"</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Empty state ───────────────────────────────────────────────────────────────
if n_nodes == 0:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:400px;flex-direction:column;gap:12px;color:#2d4060'>"
        "<div style='font-size:3rem'>🧠</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.85rem'>"
        "No memory data found. Solve some problems first!</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Render graph ──────────────────────────────────────────────────────────────
import streamlit.components.v1 as components

graph_html = render_graph_html(
    graph_data       = graph_data,
    layout_preset    = layout_choice,
    show_labels      = show_labels,
    show_edge_labels = show_edge_labels,
)

components.html(graph_html, height=720, scrolling=False)

# ── Keyboard shortcut hint ────────────────────────────────────────────────────
st.markdown(
    "<div style='font-family:JetBrains Mono,monospace;font-size:0.68rem;"
    "color:#1e3a5a;text-align:center;margin-top:4px'>"
    "⌨ &nbsp;F = fit &nbsp;· &nbsp;L = toggle labels &nbsp;· &nbsp;"
    "Esc = close panel &nbsp;· &nbsp;Double-click = expand neighbours"
    "</div>",
    unsafe_allow_html=True,
)