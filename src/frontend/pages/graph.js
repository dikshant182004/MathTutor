/*
  graph.js — vis.js network initialization and interaction logic
  frontend/graph_viz/graph.js

  Tokens injected by memory_viz.py at render time:
    %%GRAPH_JSON%%       — JSON {nodes, edges}
    %%LAYOUT_OPTIONS%%   — vis.js layout/physics options JSON
    %%PHYSICS_ENABLED%%  — "true" or "false"
    %%NODE_COLORS%%      — NODE_COLORS dict JSON
    %%TYPE_BADGE_STYLE%% — TYPE_BADGE_STYLE dict JSON
    %%NODE_SIZES%%       — NODE_SIZES dict JSON
    %%NODE_SHAPES%%      — NODE_SHAPES dict JSON
    %%NODE_FONT_SIZES%%  — NODE_FONT_SIZES dict JSON
    %%EDGE_CONFIG%%      — edge defaults JSON
    %%SHOW_LABELS%%      — "true" or "false"
*/

// ── Injected data ─────────────────────────────────────────────────────────────
const RAW              = %%GRAPH_JSON%%;
const NODE_COLORS      = %%NODE_COLORS%%;
const TYPE_BADGE_STYLE = %%TYPE_BADGE_STYLE%%;
const NODE_SIZES       = %%NODE_SIZES%%;
const NODE_SHAPES      = %%NODE_SHAPES%%;
const NODE_FONT_SIZES  = %%NODE_FONT_SIZES%%;
const EDGE_CFG         = %%EDGE_CONFIG%%;
let   showLabels       = %%SHOW_LABELS%%;

// ── Build vis.js datasets ─────────────────────────────────────────────────────
function buildVisNode(n) {
  const c    = NODE_COLORS[n.type] || NODE_COLORS["agent"];
  const size = NODE_SIZES[n.type]  || 18;
  const fs   = NODE_FONT_SIZES[n.type] || 11;
  return {
    id:    n.id,
    label: showLabels ? n.label.replace(/\\n/g, "\n") : "",
    title: n.title || n.label,
    color: {
      background: c.background,
      border:     c.border,
      highlight:  { background: c.highlight_background, border: c.highlight_border },
      hover:      { background: c.hover_background,     border: c.hover_border },
    },
    size:   size,
    font:   {
      color: showLabels ? "#c8d8f0" : "transparent",
      size:  fs,
      face:  "JetBrains Mono, Fira Code, monospace",
      strokeWidth: showLabels ? 2 : 0,
      strokeColor: "#0a0d14",
    },
    shape:  NODE_SHAPES[n.type] || "dot",
    shadow: { enabled: true, color: "rgba(0,0,0,0.6)", size: 10, x: 3, y: 3 },
    _type:  n.type,
  };
}

const visNodes = RAW.nodes.map(buildVisNode);

const visEdges = RAW.edges.map((e, i) => ({
  id:     "e" + i,
  from:   e.from,
  to:     e.to,
  label:  e.label || "",
  arrows: e.arrows || "to",
  dashes: e.dashes || false,
  color:  { color: EDGE_CFG.color, highlight: EDGE_CFG.highlight, hover: EDGE_CFG.hover },
  font:   { color: EDGE_CFG.font_color, size: EDGE_CFG.font_size, align: "middle",
            strokeWidth: 2, strokeColor: "#0a0d14" },
  width:  EDGE_CFG.width,
  smooth: { type: EDGE_CFG.smooth_type, forceDirection: "none", roundness: EDGE_CFG.smooth_round },
}));

// ── Network init ──────────────────────────────────────────────────────────────
const container = document.getElementById("mynetwork");
const nodeDS    = new vis.DataSet(visNodes);
const edgeDS    = new vis.DataSet(visEdges);

const layoutOpts = %%LAYOUT_OPTIONS%%;
const options = Object.assign({}, layoutOpts, {
  interaction: {
    hover:             true,
    tooltipDelay:      120,
    navigationButtons: false,
    keyboard:          { enabled: true, bindToWindow: false },
    multiselect:       true,
    zoomSpeed:         0.6,
  },
  nodes: { borderWidth: 1.5, borderWidthSelected: 2.5 },
  edges: { selectionWidth: 3 },
});

const network = new vis.Network(container, { nodes: nodeDS, edges: edgeDS }, options);

// ── Stats bar ─────────────────────────────────────────────────────────────────
const statNodes    = document.getElementById("stat-nodes");
const statEdges    = document.getElementById("stat-edges");
const statSelected = document.getElementById("stat-selected");

function updateStats(selLabel) {
  statNodes.textContent    = RAW.nodes.length + " nodes";
  statEdges.textContent    = RAW.edges.length + " edges";
  statSelected.textContent = selLabel || "Nothing selected";
}
updateStats();

// ── Physics toggle ────────────────────────────────────────────────────────────
let physicsEnabled = %%PHYSICS_ENABLED%%;

function togglePhysics() {
  physicsEnabled = !physicsEnabled;
  network.setOptions({ physics: { enabled: physicsEnabled } });
  const btn = document.getElementById("btn-physics");
  btn.classList.toggle("active", physicsEnabled);
  btn.textContent = physicsEnabled ? "⚡ Physics" : "❄ Frozen";
}

// ── Label toggle ──────────────────────────────────────────────────────────────
// Exposed globally so Streamlit parent can call it via postMessage if needed
function toggleLabels() {
  showLabels = !showLabels;
  const updates = nodeDS.get().map(n => {
    const raw = RAW.nodes.find(r => r.id === n.id);
    return {
      id:    n.id,
      label: showLabels ? (raw ? raw.label.replace(/\\n/g, "\n") : "") : "",
      font:  {
        color: showLabels ? "#c8d8f0" : "transparent",
        strokeWidth: showLabels ? 2 : 0,
      },
    };
  });
  nodeDS.update(updates);
}

// ── Toolbar actions ───────────────────────────────────────────────────────────
function fitGraph() { network.fit({ animation: { duration: 600, easingFunction: "easeInOutCubic" } }); }
function zoomIn()   { network.moveTo({ scale: network.getScale() * 1.3, animation: { duration: 250 } }); }
function zoomOut()  { network.moveTo({ scale: network.getScale() * 0.77, animation: { duration: 250 } }); }

function exportPNG() {
  try {
    const canvas = container.querySelector("canvas");
    if (!canvas) return;
    const link    = document.createElement("a");
    link.download = "memory_graph.png";
    link.href     = canvas.toDataURL("image/png");
    link.click();
  } catch (e) { console.warn("PNG export failed:", e); }
}

// ── Panel ──────────────────────────────────────────────────────────────────────
const panel      = document.getElementById("panel");
const panelTitle = document.getElementById("panel-title");
const panelBody  = document.getElementById("panel-body");

function openPanel()  { panel.classList.add("open"); }
function closePanel() {
  panel.classList.remove("open");
  network.unselectAll();
  updateStats();
}

// ── Decay score styling ───────────────────────────────────────────────────────
function decayClass(val) {
  const v = parseFloat(val);
  if (isNaN(v)) return "";
  if (v >= 0.6) return "decay-high";
  if (v >= 0.3) return "decay-mid";
  return "decay-low";
}

// ── Node panel ────────────────────────────────────────────────────────────────
function renderNodePanel(nodeId) {
  const n = RAW.nodes.find(x => x.id === nodeId);
  if (!n) return;
  openPanel();
  panelTitle.textContent = n.label.replace(/\n/g, " ").slice(0, 40);

  const badgeStyle = TYPE_BADGE_STYLE[n.type]
    || "background:#111d30;color:#7090b0;border:1px solid #2d406055";
  let html = `<span class="type-badge" style="${badgeStyle}">${n.type}</span>`;

  const detail = n.detail || {};
  Object.entries(detail).forEach(([k, v]) => {
    if (!v || v === "—" || v === "None") return;
    const valStr = String(v).slice(0, 300);
    const cls    = k.toLowerCase().includes("decay")
      ? ` class="detail-val ${decayClass(valStr)}"` : ` class="detail-val"`;
    html += `
      <div class="detail-row">
        <div class="detail-key">${k}</div>
        <div${cls}>${valStr}</div>
      </div>`;
  });

  // Connected nodes
  const connEdges = RAW.edges.filter(e => e.from === nodeId || e.to === nodeId);
  if (connEdges.length) {
    html += `<div class="conn-section"><div class="conn-label">Connections (${connEdges.length})</div>`;
    connEdges.forEach(e => {
      const otherId = e.from === nodeId ? e.to : e.from;
      const other   = RAW.nodes.find(x => x.id === otherId);
      if (!other) return;
      const c   = NODE_COLORS[other.type] || NODE_COLORS["agent"];
      const dir = e.from === nodeId ? "→" : "←";
      html += `
        <div class="conn-item" onclick="focusNode('${otherId}')">
          <div class="conn-dot" style="background:${c.border}"></div>
          <span class="conn-dir">${dir} ${e.label || ""}&nbsp;</span>
          <span>${(other.label || "").replace(/\n/g, " ").slice(0, 28)}</span>
        </div>`;
    });
    html += `</div>`;
  }

  panelBody.innerHTML = html;
  updateStats(n.label.replace(/\n/g, " ").slice(0, 32));
}

// ── Edge panel ────────────────────────────────────────────────────────────────
function renderEdgePanel(edgeId) {
  const idx     = parseInt(edgeId.replace("e", ""), 10);
  const rawEdge = RAW.edges[idx];
  if (!rawEdge) return;
  openPanel();
  panelTitle.textContent = rawEdge.label || "Relationship";

  const fromNode = RAW.nodes.find(x => x.id === rawEdge.from);
  const toNode   = RAW.nodes.find(x => x.id === rawEdge.to);

  let html = `<span class="type-badge" style="background:#0d1a30;color:#7eb8f7;border:1px solid #3b82f633;">relationship</span>`;
  html += `<div class="detail-row"><div class="detail-key">Type</div><div class="detail-val">${rawEdge.label || "—"}</div></div>`;
  html += `<div class="detail-row"><div class="detail-key">Style</div><div class="detail-val">${rawEdge.dashes ? "dashed — derived link" : "solid — direct link"}</div></div>`;

  if (fromNode && toNode) {
    html += `<div class="conn-section"><div class="conn-label">Endpoints</div>`;
    [fromNode, toNode].forEach(node => {
      const c = NODE_COLORS[node.type] || NODE_COLORS["agent"];
      html += `
        <div class="conn-item" onclick="focusNode('${node.id}')">
          <div class="conn-dot" style="background:${c.border}"></div>
          <span>${node.label.replace(/\n/g, " ").slice(0, 28)}</span>
          <span style="color:#2d4060;font-size:9px;margin-left:auto">${node.type}</span>
        </div>`;
    });
    html += `</div>`;
  }

  panelBody.innerHTML = html;
  updateStats(`Edge: ${rawEdge.label || "rel"}`);
}

// ── Focus helper ──────────────────────────────────────────────────────────────
function focusNode(nodeId) {
  network.focus(nodeId, { scale: 1.5, animation: { duration: 450, easingFunction: "easeInOutCubic" } });
  network.selectNodes([nodeId]);
  renderNodePanel(nodeId);
}

// ── Network events ────────────────────────────────────────────────────────────
network.on("click", params => {
  if (params.nodes.length)      renderNodePanel(params.nodes[0]);
  else if (params.edges.length) renderEdgePanel(params.edges[0]);
  else                          closePanel();
});

network.on("doubleClick", params => {
  if (!params.nodes.length) return;
  const nid     = params.nodes[0];
  const connIds = RAW.edges
    .filter(e => e.from === nid || e.to === nid)
    .map(e => e.from === nid ? e.to : e.from);
  if (connIds.length) {
    network.selectNodes([nid, ...connIds]);
    network.fit({ nodes: [nid, ...connIds], animation: { duration: 500 } });
  }
});

// ── Hover glow ────────────────────────────────────────────────────────────────
network.on("hoverNode", params => {
  const n = nodeDS.get(params.node);
  if (n) nodeDS.update({
    id: params.node,
    shadow: { enabled: true, color: n.color.border, size: 22, x: 0, y: 0 },
  });
  container.style.cursor = "pointer";
});

network.on("blurNode", params => {
  const n = nodeDS.get(params.node);
  if (n) nodeDS.update({
    id: params.node,
    shadow: { enabled: true, color: "rgba(0,0,0,0.6)", size: 10, x: 3, y: 3 },
  });
  container.style.cursor = "default";
});

network.on("hoverEdge", () => { container.style.cursor = "pointer"; });
network.on("blurEdge",  () => { container.style.cursor = "default"; });

// ── Stabilisation ─────────────────────────────────────────────────────────────
network.on("stabilizationProgress", p => {
  if (p.iterations % 25 === 0) {
    const pct = Math.round(p.iterations / p.total * 100);
    statSelected.textContent = `Stabilising… ${pct}%`;
  }
});

network.on("stabilizationIterationsDone", () => {
  network.setOptions({ physics: { enabled: false } });
  physicsEnabled = false;
  const btn = document.getElementById("btn-physics");
  if (btn) { btn.textContent = "❄ Frozen"; btn.classList.remove("active"); }
  updateStats();
});

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closePanel();
  if (e.key === "f" || e.key === "F") fitGraph();
  if (e.key === "l" || e.key === "L") toggleLabels();
});