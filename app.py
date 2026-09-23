"""Local presentation of immutable Stage 1 outputs: streamlit run app.py."""

from pathlib import Path
from html import escape
from decimal import Decimal, InvalidOperation
import math
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import (Analyst, DIAGNOSTICS, FILES, PRIORITY_NOTE, load_data, role_counts, summary)

ROOT = Path(__file__).resolve().parent
GRAPH_EDGE_LIMIT = 30

# Presentation tokens only: no analytical meaning or runtime theme configuration.
COLORS = {
    "background": "#0c121b", "sidebar": "#101925", "surface": "#141f2c",
    "elevated": "#1a2939", "border": "#2c3e50", "text": "#e8eef5",
    "secondary": "#b7c6d5", "muted": "#92a7bb", "accent": "#69c9b0",
    "success": "#7bceb1", "warning": "#e7ba73", "selected": "#234b73",
    "incoming": "#65c5a3", "outgoing": "#e0b16c", "neighbor": "#93b3ca",
}
ROLE_COLORS = {
    "consolidator": "#80b9e8", "transit": "#72cbbb", "distributor": "#e7ba73",
    "terminal": "#d5a6b9", "coordinator": "#b5a4e3", "peripheral": "#a2b0c0",
}
FONT = "Segoe UI, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
STYLES = "<style>:root{" + "".join(f"--mg-{key}:{value};" for key, value in COLORS.items()) + "}" + """
html {color-scheme:dark;}
.stApp {background:var(--mg-background);color:var(--mg-text);
    font-family:'Segoe UI',system-ui,-apple-system,BlinkMacSystemFont,sans-serif;}
.stAppHeader {background:var(--mg-background);color:var(--mg-secondary);}
.stMainBlockContainer {max-width:1180px;padding:4.5rem 2.6rem 3rem;}
.stSidebar {background:var(--mg-sidebar);border-right:1px solid var(--mg-border);color:var(--mg-text);}
.stSidebar h2 {font-size:1.4rem;letter-spacing:-.035em;}
.stSidebar [role="radiogroup"] {gap:.45rem;}
.stSidebar [role="radiogroup"] label {padding:.7rem .8rem;border:1px solid transparent;
    border-radius:8px;width:100%;color:var(--mg-secondary);}
.stSidebar [role="radiogroup"] label:has(input:checked) {
    background:var(--mg-elevated);border-color:var(--mg-border);color:var(--mg-text);}
.stSidebar input[type="radio"] {accent-color:var(--mg-accent);}
.stApp h1 {color:var(--mg-text);font-weight:650;font-size:clamp(1.7rem,3vw,2.55rem);
    letter-spacing:-.045em;line-height:1.2;overflow-wrap:anywhere;padding-bottom:.35rem;}
.stApp h2 {color:var(--mg-text);font-size:1.5rem;letter-spacing:-.025em;font-weight:620;}
.stApp h3 {color:var(--mg-text);font-size:1.12rem;letter-spacing:-.01em;font-weight:620;}
.stApp p {line-height:1.6;}
.stCaption,.stCaptionContainer {color:var(--mg-muted);font-size:.88rem;}
.stApp a {color:var(--mg-accent);}
.mg-eyebrow {font-size:.73rem;font-weight:650;letter-spacing:.16em;color:var(--mg-accent);
    text-transform:uppercase;margin:0 0 .3rem;}
.mg-brand {border:1px solid var(--mg-border);border-radius:8px;width:42px;height:42px;
    display:grid;place-items:center;color:var(--mg-accent);font-size:1.1rem;font-weight:700;
    background:var(--mg-surface);margin-bottom:1rem;}
.mg-role {display:inline-block;border:1px solid currentColor;border-radius:6px;padding:.2rem .65rem;
    font-size:.83rem;font-weight:600;letter-spacing:.01em;background:var(--mg-surface);}
/* Metrics and alerts require a small number of Streamlit element hooks;
   no generated classes, DOM position selectors, or hidden controls. */
[data-testid="stMetric"] {background:var(--mg-surface);border:1px solid var(--mg-border);
    border-radius:10px;padding:1rem 1.15rem;min-height:108px;box-shadow:0 2px 8px #00000012;}
[data-testid="stMetricLabel"] {color:var(--mg-secondary);white-space:normal;}
[data-testid="stMetricLabel"] p {font-size:.85rem;line-height:1.35;}
[data-testid="stMetricValue"] {color:var(--mg-text);font-size:clamp(1.15rem,2.3vw,1.95rem);
    font-weight:620;letter-spacing:-.035em;font-variant-numeric:tabular-nums;
    white-space:normal;overflow-wrap:anywhere;line-height:1.3;}
[data-testid="stMetricValue"] > div {white-space:normal;overflow-wrap:anywhere;text-overflow:clip;}
[data-testid="stAlert"] {border-radius:8px;border:1px solid var(--mg-border);}
[data-testid="stAlert"][data-baseweb="notification"] {color:var(--mg-text);}
.stAlert p {font-size:.9rem;line-height:1.55;}
.st-key-node-explanation,.st-key-agent-explanation {background:var(--mg-surface);
    border-radius:10px;border-left:3px solid var(--mg-accent);padding:1.05rem 1.25rem;}
.st-key-node-explanation p,.st-key-agent-explanation p {color:var(--mg-secondary);font-size:.94rem;}
.st-key-node-explanation h3,.st-key-agent-explanation h3 {color:var(--mg-text);}
.stApp label {color:var(--mg-secondary);}
.stTextInput input {background:var(--mg-surface);color:var(--mg-text);caret-color:var(--mg-accent);
    font-family:Consolas,'Courier New',monospace;font-size:.94rem;}
.stTextInput input::placeholder {color:var(--mg-muted);opacity:1;}
.stTextInput [data-baseweb="input"],.stSelectbox [data-baseweb="select"] > div {
    background:var(--mg-surface);border-color:var(--mg-border);color:var(--mg-text);border-radius:8px;}
.stSelectbox input {color:var(--mg-text);}
[role="listbox"],[role="option"] {background:var(--mg-elevated);color:var(--mg-text);}
[role="option"][aria-selected="true"] {background:var(--mg-surface);color:var(--mg-accent);}
.stApp button {border-radius:8px;border-color:var(--mg-border);}
.stApp button[kind="primaryFormSubmit"],.stApp button[kind="primary"] {
    background:var(--mg-accent);color:#0c211c;border-color:var(--mg-accent);font-weight:650;}
.stApp :focus-visible {outline:2px solid var(--mg-accent);outline-offset:3px;}
.stApp details {background:var(--mg-surface);border:1px solid var(--mg-border);border-radius:8px;}
.stApp summary {color:var(--mg-secondary);font-size:.9rem;}
.stApp summary:hover {color:var(--mg-text);background:var(--mg-elevated);border-radius:8px;}
.stForm {background:var(--mg-surface);border-color:var(--mg-border);border-radius:10px;}
.stCode {border:1px solid var(--mg-border);border-radius:8px;}
.stApp pre {background:var(--mg-surface);color:var(--mg-secondary);line-height:1.65;}
.stApp code {font-family:Consolas,'Courier New',monospace;}
.stPlotlyChart {border:1px solid var(--mg-border);border-radius:10px;overflow:clip;}
.mg-legend {display:flex;flex-wrap:wrap;gap:1.4rem;align-items:center;color:var(--mg-secondary);
    font-size:.82rem;padding:.25rem 0 .6rem;}
.mg-legend span {display:inline-flex;align-items:center;gap:.5rem;}
.mg-dot {display:inline-block;width:11px;height:11px;border-radius:50%;background:var(--mg-selected);
    border:1px solid var(--mg-neighbor);}
.mg-incoming {color:var(--mg-incoming);font-size:1.1rem;}
.mg-outgoing {color:var(--mg-outgoing);font-size:1.1rem;}
.inspection-table {width:100%;table-layout:fixed;border-collapse:separate;border-spacing:0;
    font-size:.875rem;margin:0 0 1.1rem;background:var(--mg-surface);color:var(--mg-secondary);
    border:1px solid var(--mg-border);border-radius:9px;overflow:hidden;font-variant-numeric:tabular-nums;}
.inspection-table th,.inspection-table td {text-align:left!important;padding:.8rem .7rem;
    vertical-align:top;overflow-wrap:anywhere;border-bottom:1px solid var(--mg-border);line-height:1.5;}
.inspection-table th {background:var(--mg-elevated);color:var(--mg-text);font-size:.78rem;
    font-weight:600;letter-spacing:.025em;}
.inspection-table td:first-of-type {font-family:Consolas,'Courier New',monospace;color:var(--mg-text);}
.inspection-table tbody tr:hover {background:var(--mg-elevated);}
.inspection-table tbody tr:last-of-type td {border-bottom:0;}
@media(max-width:900px) {
    .stMainBlockContainer {padding:4.5rem 1.2rem 2rem;}
    [data-testid="stMetric"] {padding:.8rem;min-height:96px;}
    .inspection-table th,.inspection-table td {padding:.65rem .45rem;}
}
@media(prefers-reduced-motion:reduce) {.stApp * {scroll-behavior:auto;}}
</style>"""


@st.cache_data(show_spinner=False)
def cached_data(root, signature):
    # File metadata invalidates cache after explicit regeneration; no pipeline invocation.
    return load_data(Path(root))


def role_chart(counts):
    fig = go.Figure(go.Bar(x=list(counts.values()), y=list(counts), orientation="h",
                           marker_color=[ROLE_COLORS[role] for role in counts],
                           text=list(counts.values()), textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>Observed nodes: %{x}<extra></extra>"))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=15, r=55, t=20, b=35),
                      paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
                      font=dict(family=FONT, size=13, color=COLORS["secondary"]),
                      hoverlabel=dict(bgcolor=COLORS["elevated"], bordercolor=COLORS["border"],
                                      font=dict(family=FONT, color=COLORS["text"])),
                      bargap=.42, yaxis=dict(autorange="reversed", showgrid=False, ticks=""),
                      xaxis=dict(gridcolor=COLORS["border"], zeroline=False, tickfont=dict(size=11)),
                      xaxis_title="Observed nodes")
    return fig


def local_graph(data, gid):
    """Only incident observed edges; deterministic geometric layout, not a graph metric."""
    gid = Analyst(data).inspect_node(gid)["gid"]
    edges = data["edges"]
    incident = edges[(edges.src == gid) | (edges.dst == gid)].copy()
    if incident.empty:
        return None
    incident = incident.assign(_src=incident.src.map(int), _dst=incident.dst.map(int)).sort_values(["_src", "_dst"])
    if len(incident) > GRAPH_EDGE_LIMIT:
        incident = incident.sort_values(["sum_kzt", "n_tx", "_src", "_dst"],
                                        ascending=[False, False, True, True]).head(GRAPH_EDGE_LIMIT)
    neighbors = sorted((set(incident.src) | set(incident.dst)) - {gid}, key=int)
    positions = {gid: (0.0, 0.0)}
    positions.update({node: (math.cos(2 * math.pi * i / len(neighbors)),
                            math.sin(2 * math.pi * i / len(neighbors))) for i, node in enumerate(neighbors)})
    pairs = set(zip(incident.src, incident.dst))
    fig = go.Figure()
    edge_records = []
    for row in incident.itertuples(index=False):
        x0, y0 = positions[row.src]
        x1, y1 = positions[row.dst]
        # Opposite directed edges curve to opposite sides and retain distinct tooltips.
        offset = 0.14 if (row.dst, row.src) in pairs else 0
        cx, cy = (x0 + x1) / 2 - offset * (y1 - y0), (y0 + y1) / 2 + offset * (x1 - x0)
        points = []
        for i in range(21):
            t = i / 20
            if row.src == row.dst:
                points.append((x0 + 0.15 * math.sin(2 * math.pi * t),
                               y0 + 0.15 * (1 - math.cos(2 * math.pi * t))))
            else:
                points.append(((1-t)**2*x0 + 2*(1-t)*t*cx + t*t*x1,
                               (1-t)**2*y0 + 2*(1-t)*t*cy + t*t*y1))
        color = COLORS["incoming"] if row.dst == gid else COLORS["outgoing"]
        hover = (f"Source GID: {row.src}<br>Destination GID: {row.dst}"
                 f"<br>Amount: {format_kzt(row.sum_kzt)}<br>Transactions: {row.n_tx}")
        fig.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points],
                                mode="lines", line=dict(color=color, width=2), opacity=.9,
                                text=[hover]*len(points), hovertemplate="%{text}<extra></extra>",
                                showlegend=False, name=f"{row.src} → {row.dst}"))
        tail, head = points[-6], points[-3]
        fig.add_annotation(x=head[0], y=head[1], ax=tail[0], ay=tail[1],
                           xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=3, arrowsize=1.6, arrowwidth=2, arrowcolor=color)
        edge_records.append({"src": row.src, "dst": row.dst, "sum_kzt": row.sum_kzt, "n_tx": row.n_tx})
    nodes = data["nodes"].set_index("gid")
    ordered = neighbors + [gid]
    fig.add_trace(go.Scatter(x=[positions[g][0] for g in ordered], y=[positions[g][1] for g in ordered],
                            mode="markers", marker=dict(size=[13]*len(neighbors)+[30],
                            color=[COLORS["neighbor"]]*len(neighbors)+[COLORS["selected"]],
                            line=dict(width=[1]*len(neighbors)+[3], color=COLORS["text"])),
                            text=[f"GID: {g}<br>Role: {nodes.at[g, 'role']}<br>Review priority: {format_score(nodes.at[g, 'priority_score'])}"
                                  for g in ordered], hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.update_layout(template="plotly_dark", height=480, margin=dict(l=20, r=20, t=20, b=20),
                      xaxis=dict(visible=False, range=[-1.2, 1.2]),
                      yaxis=dict(visible=False, range=[-1.2, 1.2], scaleanchor="x", scaleratio=1),
                      hovermode="closest", paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
                      font=dict(family=FONT, color=COLORS["text"]),
                      hoverlabel=dict(bgcolor=COLORS["elevated"], bordercolor=COLORS["border"],
                                      font=dict(family=FONT, size=13, color=COLORS["text"])),
                      meta={"selected_gid": gid, "node_gids": ordered, "observed_edges": edge_records})
    return fig


LABELS = {
    "rank": "Rank", "gid": "GID", "neighbor_gid": "Neighbor GID", "role": "Role",
    "role_score": "Role score", "priority_score": "Review priority", "why": "Key signals",
    "direction": "Direction", "sum_kzt": "Amount", "n_tx": "Transactions",
    "cluster_id": "Cluster ID", "n_nodes": "Members", "n_seed": "Seed nodes",
    "sum_kzt_internal": "Internal flow", "top_gids": "Top GIDs",
    "hypothesis": "Structural hypothesis", "evidence": "Evidence",
    "in_deg": "Incoming relationships", "out_deg": "Outgoing relationships",
    "in_kzt": "Incoming amount", "out_kzt": "Outgoing amount",
    "in_tx": "Incoming transactions", "out_tx": "Outgoing transactions",
    "betweenness_pct": "Betweenness percentile", "pagerank_pct": "PageRank percentile",
    "prior_day_in_fraction": "Prior-day activity signal",
}


def display_number(value):
    """Decimal conversion for display only; never convert identifiers to floats."""
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def format_kzt(value, compact=False):
    number = display_number(value)
    if number is None:
        return "Unavailable"
    if compact:
        for scale, suffix in ((10**9, "B"), (10**6, "M"), (10**3, "K")):
            if abs(number) >= scale:
                return f"₸{number / scale:.2f}".rstrip("0").rstrip(".") + suffix
    exact = format(number, ",f")
    return "₸" + (exact.rstrip("0").rstrip(".") if "." in exact else exact)


def format_percent(value):
    number = display_number(value)
    return "Unavailable" if number is None else f"{number * 100:.2f}%"


def format_score(value):
    number = display_number(value)
    return "Unavailable" if number is None else f"{number:.4f}"


def display_frame(frame):
    """Preserve row order and original data; only the rendered copy is formatted."""
    shown = frame.copy(deep=True)
    for key in shown.columns:
        if key in ("sum_kzt", "sum_kzt_internal", "in_kzt", "out_kzt"):
            shown[key] = shown[key].map(format_kzt)
        elif key in ("role_score", "priority_score"):
            shown[key] = shown[key].map(format_score)
        elif key in ("betweenness_pct", "pagerank_pct", "prior_day_in_fraction"):
            shown[key] = shown[key].map(format_percent)
        elif key == "direction":
            shown[key] = shown[key].map(lambda value: {"incoming": "Incoming", "outgoing": "Outgoing"}.get(value, value))
    return shown.rename(columns=LABELS)


def node_summary(node):
    """Labels and stored values, without new role reasoning or missing-field inference."""
    groups = (
        ("Assigned role", ("role", "role_score", "priority_score")),
        ("Observed flow", ("in_deg", "out_deg")),
        ("Money flow", ("in_kzt", "out_kzt")),
        ("Transactions", ("in_tx", "out_tx")),
        ("Network position", ("betweenness_pct", "pagerank_pct")),
    )
    for title, keys in groups:
        values = []
        for key in keys:
            value = node.get(key)
            if value is None or pd.isna(value):
                continue
            if key in ("in_kzt", "out_kzt"):
                value = format_kzt(value, compact=True)
            elif key.endswith("_pct"):
                value = format_percent(value)
            elif key in ("role_score", "priority_score"):
                value = format_score(value)
            values.append(f"{LABELS[key]}: {value}")
        if values:
            st.write(f"**{title}** — " + " · ".join(values))


def table(frame, technical=False):
    # Wrapped, escaped HTML avoids horizontal scrolling for evidence and hypotheses.
    shown = frame if technical else display_frame(frame)
    st.html(shown.to_html(index=False, escape=True, border=0, classes="inspection-table"))


def detail_table(frame, noun):
    """Preserve retrieval order; keep the complete table available on demand."""
    st.caption(f"Showing {min(15, len(frame))} of {len(frame)} {noun}.")
    table(frame.head(15))
    if len(frame) > 15:
        with st.expander(f"Show all {len(frame)} {noun}", expanded=False):
            table(frame)


def parse_hypothesis(hypothesis):
    """Extract unique, exact stored fields only; malformed/ambiguous fields are omitted."""
    if not isinstance(hypothesis, str):
        return {}
    fields = {}
    for part in hypothesis.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator:
            fields.setdefault(key, []).append(value)
    formats = {
        "category": ("Structural category", r"[a-z]+(?:[-/][a-z]+)*"),
        "dominant_role": ("Dominant role", r"consolidator|transit|distributor|terminal|coordinator|peripheral"),
        "dominant_share": ("Dominant share", r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)"),
        "internal": ("Internal KZT", r"[0-9]+(?:\.[0-9]+)? KZT"),
        "truncated": ("Truncated nodes", r"[0-9]+"),
    }
    result = {}
    for key, (label, pattern) in formats.items():
        values = fields.get(key, [])
        if len(values) == 1 and re.fullmatch(pattern, values[0]):
            result[label] = values[0]
    return result


def structural_summary(hypothesis, internal_kzt=None):
    fields = list(parse_hypothesis(hypothesis).items())
    for start in range(0, len(fields), 3):
        for col, (label, value) in zip(st.columns(3), fields[start:start+3]):
            if label == "Dominant share":
                value = format_percent(value)
            elif label == "Internal KZT":
                value = format_kzt(internal_kzt if internal_kzt is not None else value.removesuffix(" KZT"), compact=True)
            col.metric(label, value)
    with st.expander("Full stored structural hypothesis", expanded=False):
        st.write(hypothesis)


def show_node(data):
    st.header("Node Inspector")
    options = sorted(data["nodes"].gid.tolist(), key=int)
    select_column, exact_column = st.columns(2)
    selected = select_column.selectbox("Select gid from all nodes", options, index=None, key="node_gid")
    exact = exact_column.text_input("Or enter exact gid", key="exact_gid", placeholder="Paste the full integer gid")
    gid = exact.strip() or selected
    if gid is None:
        st.info("Choose a node to inspect its role, observed relationships and limitations.")
        return
    agent = Analyst(data)
    result = agent.dispatch("inspect_node", gid)
    if not result["ok"]:
        st.error(result["error"])
        return
    node = result["data"]
    st.subheader(f"GID {node['gid']}")
    st.html(f'<span class="mg-role" style="color:{ROLE_COLORS.get(node["role"], COLORS["secondary"])}">'
            f'{escape(node["role"])}</span>')
    cols = st.columns(3)
    cols[0].metric("Role", node["role"])
    cols[1].metric("Role score", f"{node['role_score']:.4f}")
    cols[2].metric("Review priority", f"{node['priority_score']:.4f}")
    st.caption(f"Cluster ID: {node['cluster_id']} · Depth: {node['depth']} · "
               f"Seed node: {node['is_seed']} · Truncated by depth: {node['truncated_by_depth']}")
    explanation = agent.explain_node(gid)
    with st.container(key="node-explanation"):
        st.subheader("Why this role?")
        node_summary(node)
        st.caption("Technical evidence")
        st.write(explanation["explanation"])
    for warning in explanation["limitations"]:
        st.warning(warning)
    diagnostics = {key: node[key] for key in DIAGNOSTICS if key in node}
    if "pass_through" in diagnostics and (diagnostics["pass_through"] == -1 or
                                          diagnostics.get("pass_through_defined") is False):
        diagnostics["pass_through"] = "Undefined (stored sentinel -1); not a zero ratio"
    with st.expander("Stored diagnostic fields"):
        table(pd.DataFrame([{"field": key, "value": str(value)} for key, value in diagnostics.items()]), technical=True)
    st.subheader("Observed local network")
    neighbors = agent.get_neighbors(gid)
    for col, label, value in zip(st.columns(3),
                                ["Incoming relationships", "Outgoing relationships", "Observed incident edges"],
                                [sum(r["direction"] == "incoming" for r in neighbors),
                                 sum(r["direction"] == "outgoing" for r in neighbors), len(neighbors)]):
        col.metric(label, value)
    if not neighbors:
        st.info("No observed incoming or outgoing edges.")
    else:
        st.caption("Selected node: large navy marker. Green: incoming. Amber: outgoing. "
                   "Arrowheads show direction; hover for exact gids and edge amounts. Layout has no analytical meaning.")
        if len(neighbors) > GRAPH_EDGE_LIMIT:
            st.caption(f"Showing {GRAPH_EDGE_LIMIT} of {len(neighbors)} observed incident edges in the graph. "
                       f"All {len(neighbors)} observed relationships remain available below.")
        st.html('<div class="mg-legend"><span><i class="mg-dot"></i>Selected</span>'
                '<span><b class="mg-incoming">→</b>Incoming</span>'
                '<span><b class="mg-outgoing">→</b>Outgoing</span></div>')
        st.plotly_chart(local_graph(data, gid), width="stretch", key="local_graph", theme=None)
        detail_table(pd.DataFrame(neighbors), "observed relationships")
    st.subheader(f"Cluster {node['cluster_id']}")
    cluster = agent.inspect_cluster(node["cluster_id"])["cluster"]
    st.write(cluster["hypothesis"])
    st.caption("Use Cluster Inspector to view all members of this cluster.")


def show_cluster(data):
    st.header("Cluster Inspector")
    select_column, exact_column = st.columns(2)
    selected = select_column.selectbox("Select cluster_id", sorted(data["clusters"].cluster_id.tolist()),
                            index=None, key="cluster_id")
    exact = exact_column.text_input("Or enter exact cluster_id", key="exact_cluster")
    cid = exact.strip() or selected
    if cid is None:
        st.info("Choose a cluster to see its stored hypothesis and members.")
        return
    result = Analyst(data).dispatch("inspect_cluster", cid)
    if not result["ok"]:
        st.error(result["error"])
        return
    detail = result["data"]
    cluster = detail["cluster"]
    st.html('<p class="mg-eyebrow">Cluster profile</p>')
    st.subheader(f"Cluster {cluster['cluster_id']}")
    structural_summary(cluster["hypothesis"], cluster["sum_kzt_internal"])
    table(pd.DataFrame([cluster])[["n_nodes", "n_seed", "sum_kzt_internal", "top_gids"]])
    st.plotly_chart(role_chart(detail["role_distribution"]), width="stretch", theme=None)
    st.caption("Members: stored priority descending, exact gid ascending.")
    detail_table(pd.DataFrame(detail["members"])[["gid", "role", "priority_score", "evidence"]], "members")


def show_agent(data):
    st.header("Deterministic Agentic Analyst")
    st.caption("Select an explicit action. The audit trace records executed retrieval operations.")
    with st.form("agent_action"):
        action_column, parameter_column = st.columns(2)
        action = action_column.selectbox("Action", ["inspect_node", "inspect_cluster", "get_top_priority", "get_neighbors", "explain_node"])
        parameter = parameter_column.text_input("Parameter: exact gid, cluster_id or n")
        run = st.form_submit_button("Run action", type="primary")
    if run:
        response = Analyst(data).dispatch(action, parameter)
        if response["ok"]:
            st.success("Action completed")
            result = response["data"]
            if action == "explain_node":
                with st.container(key="agent-explanation"):
                    st.subheader("Why this role?")
                    stored = data["nodes"].loc[data["nodes"].gid == result["gid"]].to_dict("records")[0]
                    node_summary(stored)
                    st.caption("Technical evidence")
                    st.write(result["explanation"])
                for warning in result["limitations"]:
                    st.warning(warning)
            elif action == "inspect_node":
                table(pd.DataFrame([result])[["gid", "role", "role_score", "priority_score", "evidence"]])
            elif action == "inspect_cluster":
                cluster = result["cluster"]
                table(pd.DataFrame([cluster])[["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal"]])
                structural_summary(cluster["hypothesis"], cluster["sum_kzt_internal"])
            elif action == "get_top_priority":
                table(pd.DataFrame(result))
            elif result:
                detail_table(pd.DataFrame(result), "observed relationships")
            else:
                st.info("No observed incoming or outgoing edges.")
            with st.expander("Raw structured result", expanded=False):
                st.json(result)
        else:
            st.error(response["error"])
        st.subheader("Audit trace")
        st.code("\n".join(response["trace"]), language=None)
    else:
        st.info("Choose an action and enter its parameter. No action has run yet.")


def main():
    st.set_page_config(page_title="Money Graph | HackAlem", page_icon="↗", layout="wide")
    st.html(STYLES)
    st.html('<p class="mg-eyebrow">HackAlem / Network intelligence</p>')
    st.title("Money Graph / Граф денег")
    st.caption("Observed financial structure reconstructed from the provided transaction network.")
    st.info(PRIORITY_NOTE)
    try:
        signature = tuple((path, (ROOT/path).stat().st_mtime_ns, (ROOT/path).stat().st_size)
                          for path in FILES.values() if (ROOT/path).is_file())
        with st.spinner("Loading verified presentation data…"):
            data = cached_data(str(ROOT), signature)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        st.stop()
    st.sidebar.html('<div class="mg-brand" aria-hidden="true">MG</div>'
                    '<p class="mg-eyebrow">Money Graph</p>')
    st.sidebar.subheader("Network intelligence")
    st.sidebar.caption("Observed structure. Explained decisions.")
    st.sidebar.divider()
    page = st.sidebar.radio("View", ["Dashboard", "Node Inspector", "Cluster Inspector", "Agentic Analyst"], key="page")
    st.sidebar.divider()
    st.sidebar.caption("HackAlem · Stage 2\n\nRead-only inspection of Stage 1 outputs.")
    if page == "Dashboard":
        st.subheader("Overview")
        metrics = list(summary(data).items())
        for start in (0, 3):
            for column, (name, value) in zip(st.columns(3), metrics[start:start+3]):
                column.metric(name, f"{value:,}")
        st.subheader("Role distribution")
        st.plotly_chart(role_chart(role_counts(data["nodes"])), width="stretch", theme=None)
        st.subheader("Top priority")
        st.caption("Stored shortlist. Copy a gid into Node Inspector to follow its observed connections.")
        top = data["top"]
        st.caption(f"Showing top {min(10, len(top))} of {len(top)} priority nodes.")
        table(top.head(10)[["rank", "gid", "role", "priority_score", "why"]])
        with st.expander("Show all priority nodes", expanded=False):
            table(top)
        st.subheader("Clusters")
        clusters = data["clusters"]
        preview = clusters.sort_values(["n_nodes", "cluster_id"], ascending=[False, True]).head(10)
        st.caption(f"Showing {len(preview)} of {len(clusters)} clusters.")
        table(preview[["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "hypothesis"]])
        with st.expander("Show all clusters", expanded=False):
            table(clusters)
    elif page == "Node Inspector":
        show_node(data)
    elif page == "Cluster Inspector":
        show_cluster(data)
    else:
        show_agent(data)
    with st.expander("Observation limits"):
        st.write("Seed incoming flows may be incomplete. Collection ends at depth 4: real downstream is unknown. "
                 "Missing outgoing edges do not establish retention or a known non-terminal status. "
                 "Dates have daily precision; prior-day activity is temporal proximity, not proof of funding. "
                 "Roles and community hypotheses describe observed structure only.")


if __name__ == "__main__":
    main()
