"""Local presentation of immutable Stage 1 outputs: streamlit run app.py."""

from pathlib import Path
import math
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent import (Analyst, DIAGNOSTICS, FILES, PRIORITY_NOTE, load_data, role_counts, summary)

ROOT = Path(__file__).resolve().parent
GRAPH_EDGE_LIMIT = 30


@st.cache_data(show_spinner=False)
def cached_data(root, signature):
    # File metadata invalidates cache after explicit regeneration; no pipeline invocation.
    return load_data(Path(root))


def role_chart(counts):
    fig = go.Figure(go.Bar(x=list(counts.values()), y=list(counts), orientation="h",
                           marker_color="#15877c", text=list(counts.values()), textposition="auto"))
    fig.update_layout(height=290, margin=dict(l=5, r=15, t=5, b=10),
                      yaxis=dict(autorange="reversed"), xaxis_title="Observed nodes")
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
        color = "#15877c" if row.dst == gid else "#c77c20"
        hover = f"{row.src} → {row.dst}<br>sum_kzt={row.sum_kzt:,.2f}<br>n_tx={row.n_tx}"
        fig.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points],
                                mode="lines", line=dict(color=color, width=1.6),
                                text=[hover]*len(points), hovertemplate="%{text}<extra></extra>",
                                showlegend=False, name=f"{row.src} → {row.dst}"))
        tail, head = points[-6], points[-3]
        fig.add_annotation(x=head[0], y=head[1], ax=tail[0], ay=tail[1],
                           xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                           arrowhead=3, arrowsize=1.4, arrowwidth=1.6, arrowcolor=color)
        edge_records.append({"src": row.src, "dst": row.dst, "sum_kzt": row.sum_kzt, "n_tx": row.n_tx})
    nodes = data["nodes"].set_index("gid")
    ordered = neighbors + [gid]
    fig.add_trace(go.Scatter(x=[positions[g][0] for g in ordered], y=[positions[g][1] for g in ordered],
                            mode="markers", marker=dict(size=[11]*len(neighbors)+[24],
                            color=["#7998a3"]*len(neighbors)+["#173b57"], line=dict(width=1, color="white")),
                            text=[f"gid={g}<br>role={nodes.at[g, 'role']}<br>priority_score={nodes.at[g, 'priority_score']:.10f}"
                                  for g in ordered], hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False, range=[-1.2, 1.2]),
                      yaxis=dict(visible=False, range=[-1.2, 1.2], scaleanchor="x", scaleratio=1),
                      hovermode="closest", plot_bgcolor="#f5f8fa",
                      meta={"selected_gid": gid, "node_gids": ordered, "observed_edges": edge_records})
    return fig


def table(frame):
    # Wrapped, escaped HTML avoids horizontal scrolling for evidence and hypotheses.
    st.html(frame.to_html(index=False, escape=True, border=0, classes="inspection-table"))


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


def structural_summary(hypothesis):
    fields = list(parse_hypothesis(hypothesis).items())
    for start in range(0, len(fields), 3):
        for col, (label, value) in zip(st.columns(3), fields[start:start+3]):
            col.metric(label, value)
    with st.expander("Full stored structural hypothesis", expanded=False):
        st.write(hypothesis)


def show_node(data):
    st.header("Node Inspector")
    options = sorted(data["nodes"].gid.tolist(), key=int)
    selected = st.selectbox("Select gid from all nodes", options, index=None, key="node_gid")
    exact = st.text_input("Or enter exact gid", key="exact_gid", placeholder="Paste the full integer gid")
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
    st.subheader(f"gid {node['gid']}")
    cols = st.columns(3)
    cols[0].metric("Role", node["role"])
    cols[1].metric("Role score", f"{node['role_score']:.4f}")
    cols[2].metric("Review priority", f"{node['priority_score']:.4f}")
    st.caption(f"cluster_id={node['cluster_id']} · depth={node['depth']} · "
               f"is_seed={node['is_seed']} · truncated_by_depth={node['truncated_by_depth']}")
    explanation = agent.explain_node(gid)
    st.write(explanation["explanation"])
    for warning in explanation["limitations"]:
        st.warning(warning)
    diagnostics = {key: node[key] for key in DIAGNOSTICS if key in node}
    if "pass_through" in diagnostics and (diagnostics["pass_through"] == -1 or
                                          diagnostics.get("pass_through_defined") is False):
        diagnostics["pass_through"] = "Undefined (stored sentinel -1); not a zero ratio"
    with st.expander("Stored diagnostic fields"):
        table(pd.DataFrame([{"field": key, "value": str(value)} for key, value in diagnostics.items()]))
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
        st.plotly_chart(local_graph(data, gid), width="stretch", key="local_graph")
        detail_table(pd.DataFrame(neighbors), "observed relationships")
    st.subheader(f"Cluster {node['cluster_id']}")
    cluster = agent.inspect_cluster(node["cluster_id"])["cluster"]
    st.write(cluster["hypothesis"])
    st.caption("Use Cluster Inspector to view all members of this cluster.")


def show_cluster(data):
    st.header("Cluster Inspector")
    selected = st.selectbox("Select cluster_id", sorted(data["clusters"].cluster_id.tolist()),
                            index=None, key="cluster_id")
    exact = st.text_input("Or enter exact cluster_id", key="exact_cluster")
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
    st.subheader(f"Cluster {cluster['cluster_id']}")
    structural_summary(cluster["hypothesis"])
    table(pd.DataFrame([cluster])[["n_nodes", "n_seed", "sum_kzt_internal", "top_gids"]])
    st.plotly_chart(role_chart(detail["role_distribution"]), width="stretch")
    st.caption("Members: stored priority descending, exact gid ascending.")
    detail_table(pd.DataFrame(detail["members"])[["gid", "role", "priority_score", "evidence"]], "members")


def show_agent(data):
    st.header("Deterministic Agentic Analyst")
    st.caption("Select an explicit action. The audit trace records executed retrieval operations.")
    with st.form("agent_action"):
        action = st.selectbox("Action", ["inspect_node", "inspect_cluster", "get_top_priority", "get_neighbors", "explain_node"])
        parameter = st.text_input("Parameter: exact gid, cluster_id or n")
        run = st.form_submit_button("Run action")
    if run:
        response = Analyst(data).dispatch(action, parameter)
        if response["ok"]:
            st.success("Action completed")
            result = response["data"]
            if action == "explain_node":
                st.write(result["explanation"])
                for warning in result["limitations"]:
                    st.warning(warning)
            elif action == "inspect_node":
                table(pd.DataFrame([result])[["gid", "role", "role_score", "priority_score", "evidence"]])
            elif action == "inspect_cluster":
                cluster = result["cluster"]
                table(pd.DataFrame([cluster])[["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal"]])
                structural_summary(cluster["hypothesis"])
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
    st.html("""<style>
    .stMainBlockContainer {max-width:1100px;padding-top:2rem;}
    .inspection-table {width:100%;table-layout:fixed;border-collapse:collapse;font-size:.85rem;margin:0 0 1.2rem;}
    .inspection-table th,.inspection-table td {text-align:left!important;padding:.65rem;vertical-align:top;
      overflow-wrap:anywhere;border-bottom:1px solid #dce4e9;}
    .inspection-table th {color:#54717e;font-weight:600;}
    </style>""")
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
    page = st.sidebar.radio("View", ["Dashboard", "Node Inspector", "Cluster Inspector", "Agentic Analyst"], key="page")
    st.sidebar.caption("HackAlem · Stage 2\n\nRead-only inspection of Stage 1 outputs.")
    if page == "Dashboard":
        metrics = list(summary(data).items())
        for start in (0, 3):
            for column, (name, value) in zip(st.columns(3), metrics[start:start+3]):
                column.metric(name, f"{value:,}")
        st.subheader("Role distribution")
        st.plotly_chart(role_chart(role_counts(data["nodes"])), width="stretch")
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
