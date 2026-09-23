"""Детерминированное аналитическое ядро этапа 1. Формулы описаны в README.md."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SEED = 42
PRECISION = 10
TOP_N = 30
ROLE_CATEGORIES = {
    "consolidator": "collection-oriented",
    "transit": "transit-oriented",
    "distributor": "distribution-oriented",
    "terminal": "terminal/retention-oriented",
    "coordinator": "coordination/bridging-oriented",
    "peripheral": "peripheral/isolated",
}


def percentile(values):
    """Нулю — 0; положительным — midrank/N; константе — 0 (нет различий)."""
    values = pd.Series(values, copy=True)
    assert np.isfinite(values).all() and (values >= 0).all()
    result = pd.Series(0.0, index=values.index)
    if values.nunique() <= 1:
        return result
    positive = values > 0
    result.loc[positive] = (values[positive].rank(method="average") - 0.5) / positive.sum()
    return result


def temporal_features(df, tx):
    # В фактической схеме только даты. Операции того же дня не упорядочиваем.
    events = pd.concat([tx[["src", "date"]].rename(columns={"src": "gid"}),
                        tx[["dst", "date"]].rename(columns={"dst": "gid"})])
    activity = events.groupby("gid").date.agg(["nunique", "min", "max"])
    df["active_days"] = df.gid.map(activity["nunique"]).fillna(0).astype(int)
    span = (activity["max"] - activity["min"]).dt.days + 1
    df["activity_span_days"] = df.gid.map(span).fillna(0).astype(int)
    incoming = tx[["dst", "date"]].drop_duplicates().rename(columns={"dst": "src"})
    incoming["date"] += pd.Timedelta(days=1)
    outgoing = tx[["src", "date"]].merge(incoming.assign(prior_day=True),
                                         how="left", on=["src", "date"],
                                         validate="many_to_one")
    count = outgoing.assign(hit=outgoing.prior_day.eq(True)).groupby("src").hit.sum()
    df["out_after_prior_day_in_tx"] = df.gid.map(count).fillna(0).astype(int)
    df["prior_day_in_fraction"] = (df.out_after_prior_day_in_tx /
                                   df.out_tx.where(df.out_tx > 0, 1))


def assign_roles(df, roles):
    # Явный порядок ROLES — tie-break. np.argmax выбирает первый максимум.
    inc = sum(percentile(df[c]) for c in ("in_deg", "in_kzt", "in_tx")) / 3
    out = sum(percentile(df[c]) for c in ("out_deg", "out_kzt", "out_tx")) / 3
    degree_sum = (df.in_deg + df.out_deg).clip(lower=1)
    in_share = df.in_deg / degree_sum
    out_share = df.out_deg / degree_sum
    both = (df.in_deg > 0) & (df.out_deg > 0)
    # undefined/seed pass-through никогда не используется как реальный ноль.
    comparable = df.pass_through_defined & ~df.is_seed & both
    closeness = pd.Series(0.0, index=df.index)
    ratio = df.loc[comparable, "pass_through"]
    closeness.loc[comparable] = np.minimum(ratio, 1 / ratio)
    temporal = df.prior_day_in_fraction
    # Для seed нет коэффициента flow ratio; используется только наблюдаемая структура.
    transit = np.minimum(inc, out) * (0.65 + 0.25 * closeness + 0.10 * temporal)
    transit.loc[df.is_seed] = (np.minimum(inc, out) * (0.90 + 0.10 * temporal))[df.is_seed]
    terminal_gate = (df.in_deg > 0) & ~df.truncated_by_depth
    # Для seed terminal допускается только при полном отсутствии наблюдаемого выхода.
    retention = pd.Series(0.0, index=df.index)
    nonseed = df.pass_through_defined & ~df.is_seed
    retention.loc[nonseed] = (1 - df.loc[nonseed, "pass_through"]).clip(0, 1)
    retention.loc[df.is_seed & (df.out_deg == 0)] = 1.0
    scores = {
        "consolidator": (df.in_deg >= 2) * inc * (0.5 + 0.5 * in_share),
        "transit": both * transit,
        "distributor": (df.out_deg >= 2) * out * (0.5 + 0.5 * out_share),
        "terminal": terminal_gate * inc * retention,
        "coordinator": both * (0.70 * df.betweenness_pct + 0.30 * df.pagerank_pct)
                       * (df.betweenness > 0),
    }
    strongest = pd.DataFrame(scores).max(axis=1)
    scores["peripheral"] = 1 - strongest
    for role in roles:
        df["score_" + role] = scores[role].clip(0, 1).round(PRECISION)
    matrix = df[["score_" + role for role in roles]].to_numpy()
    df["role"] = np.asarray(roles)[matrix.argmax(axis=1)]
    df["role_score"] = matrix.max(axis=1)


def cluster_graph(G):
    # Направления суммируются ТОЛЬКО для community detection.
    U = nx.Graph()
    U.add_nodes_from(sorted(G))
    for src, dst, data in sorted(G.edges(data=True), key=lambda e: (e[0], e[1])):
        previous = U.get_edge_data(src, dst, {}).get("weight", 0.0)
        U.add_edge(src, dst, weight=previous + data["sum_kzt"])
    isolates = sorted(nx.isolates(U))
    connected = U.subgraph(sorted(set(U) - set(isolates))).copy()
    communities = (nx.community.louvain_communities(connected, seed=SEED, weight="weight")
                   if connected.number_of_edges() else [])
    groups = sorted([sorted(c) for c in communities] + [[gid] for gid in isolates],
                    key=lambda c: c[0])
    return {gid: cid for cid, group in enumerate(groups) for gid in group}


def evidence(row):
    base = (f"in_deg={row.in_deg}; out_deg={row.out_deg}; "
            f"in={row.in_kzt:.6g}; out={row.out_kzt:.6g} KZT; "
            f"tx={row.in_tx}/{row.out_tx}")
    if row.role == "coordinator":
        detail = f"; bet_pct={row.betweenness_pct:.3f}; pr_pct={row.pagerank_pct:.3f}"
    elif row.role in ("terminal", "transit"):
        ratio = (f"{row.pass_through:.3g}" if row.pass_through_defined and not row.is_seed
                 else "unavailable")
        detail = f"; out/in={ratio}; prev_day={row.prior_day_in_fraction:.3f}"
    elif row.in_deg + row.out_deg == 0:
        detail = "; observed_edges=0"
    else:
        detail = f"; role_score={row.role_score:.3f}"
    return (base + detail + f"; depth={row.depth}; truncated={int(row.truncated_by_depth)}"
            + f"; is_seed={int(row.is_seed)}")


def cluster_hypothesis(group, internal_kzt, roles):
    """Ориентация по модальной назначенной роли; равенства разрешает ROLES order."""
    counts = group.role.value_counts().reindex(roles, fill_value=0)
    shares = counts / len(group)
    dominant = max(roles, key=lambda role: int(counts[role]))
    isolated = len(group) == 1 and bool(((group.in_deg == 0) & (group.out_deg == 0)).all())
    category = "peripheral/isolated" if isolated else ROLE_CATEGORIES[dominant]
    distribution = ", ".join(f"{role}={int(counts[role])}" for role in roles)
    return (f"category={category}; dominant_role={dominant}; "
            f"dominant_share={shares[dominant]:.10f}; roles: {distribution}; "
            f"internal={internal_kzt:.2f} KZT; truncated={int(group.truncated_by_depth.sum())}; "
            "structural hypothesis only")


def build_outputs(G, df, tx, roles):
    df = df.copy()
    assert set(G) == set(df.gid)
    df["pass_through_defined"] = df.in_kzt > 0
    temporal_features(df, tx)
    # Exact directed unweighted betweenness, один раз; KZT — сила, не расстояние.
    bet = nx.betweenness_centrality(G, normalized=True, weight=None)
    df["betweenness"] = df.gid.map(bet)
    df["betweenness_pct"] = percentile(df.betweenness)
    df["pagerank_pct"] = percentile(df.pagerank)
    assign_roles(df, roles)
    df["cluster_id"] = df.gid.map(cluster_graph(G)).astype(int)
    # Самостоятельная формула приоритета, не использующая назначенную роль.
    df["volume_pct"] = percentile(df.in_kzt + df.out_kzt)
    df["tx_pct"] = percentile(df.in_tx + df.out_tx)
    df["temporal_priority"] = df.prior_day_in_fraction * df.tx_pct
    df["priority_score"] = (0.35 * df.betweenness_pct + 0.15 * df.pagerank_pct
                            + 0.25 * df.volume_pct + 0.15 * df.tx_pct
                            + 0.10 * df.temporal_priority)
    isolated = (df.in_deg + df.out_deg) == 0
    df.loc[isolated, "priority_score"] = 0.0
    df["priority_score"] = df.priority_score.round(PRECISION)
    df["evidence"] = [evidence(r) for r in df.itertuples(index=False)]
    # Только ПОСЛЕ inference: технический sentinel -1 + явная маска.
    df["pass_through"] = df.pass_through.fillna(-1.0)
    floats = df.select_dtypes(include="floating").columns
    df[floats] = df[floats].round(PRECISION)
    required = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
    df = df[required + [c for c in df.columns if c not in required]].sort_values("gid")

    internal = dict.fromkeys(sorted(df.cluster_id.unique()), 0.0)
    membership = df.set_index("gid").cluster_id.to_dict()
    for src, dst, data in G.edges(data=True):
        if membership[src] == membership[dst]:
            internal[membership[src]] += data["sum_kzt"]
    clusters = []
    for cid, group in df.groupby("cluster_id", sort=True):
        # Структурное значение: betweenness, затем PageRank, затем точный int gid.
        leaders = group.sort_values(["betweenness", "pagerank", "gid"],
                                   ascending=[False, False, True]).gid.head(5).tolist()
        hypothesis = cluster_hypothesis(group, internal[cid], roles)
        clusters.append({"cluster_id": cid, "n_nodes": len(group),
                         "n_seed": int(group.is_seed.sum()), "sum_kzt_internal": internal[cid],
                         "top_gids": json.dumps(leaders), "hypothesis": hypothesis})
    clusters = pd.DataFrame(clusters)
    top = df.sort_values(["priority_score", "gid"], ascending=[False, True]).head(TOP_N)
    why = [f"bet_pct={r.betweenness_pct:.3f}; pr_pct={r.pagerank_pct:.3f}; "
           f"volume_pct={r.volume_pct:.3f}; tx_pct={r.tx_pct:.3f}; "
           f"prev_day={r.prior_day_in_fraction:.3f}; temporal={r.temporal_priority:.3f}"
           for r in top.itertuples(index=False)]
    top = top[["gid", "role", "priority_score"]].copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    top["why"] = why
    return df.reset_index(drop=True), clusters, top.reset_index(drop=True)


def validate_outputs(df, clusters, top, nodes, edges, roles):
    """Полный CHECK, включая повторный независимый подсчёт внутренних потоков."""
    for table in (df, clusters, top):
        assert not table.isna().any().any(), "NaN в выгрузке"
        numeric = table.select_dtypes(include="number").to_numpy()
        assert not np.iscomplexobj(numeric), "Complex в выгрузке"
        assert np.isfinite(numeric).all(), "Inf в выгрузке"
    assert len(df) == len(nodes) and df.gid.is_unique
    assert pd.api.types.is_integer_dtype(df.gid), "gid нельзя округлять до float"
    assert set(df.gid) == set(nodes.gid)
    assert df.role.isin(roles).all()
    for column in ["role_score", "priority_score"] + ["score_" + r for r in roles]:
        assert df[column].between(0, 1).all(), column
    scores = df[["score_" + r for r in roles]].to_numpy()
    assert (df.role.to_numpy() == np.asarray(roles)[scores.argmax(axis=1)]).all()
    assert np.allclose(df.role_score, scores.max(axis=1), rtol=0, atol=1e-10)
    assert df.evidence.str.len().between(1, 200).all()
    assert df.evidence.str.contains(r"\d").all()
    assert (df.cluster_id >= 0).all() and (df.cluster_id % 1 == 0).all()
    assert clusters.cluster_id.is_unique
    assert set(clusters.cluster_id) == set(df.cluster_id)
    assert sorted(clusters.cluster_id) == list(range(len(clusters)))
    assert clusters.hypothesis.str.strip().str.len().gt(0).all()
    membership = df.set_index("gid").cluster_id
    inside = edges.assign(src_cluster=edges.src.map(membership), dst_cluster=edges.dst.map(membership))
    amounts = inside[inside.src_cluster == inside.dst_cluster].groupby("src_cluster").sum_kzt.sum()
    for row in clusters.itertuples(index=False):
        group = df[df.cluster_id == row.cluster_id]
        assert row.n_nodes == len(group)
        assert row.n_seed == int(nodes[nodes.gid.isin(group.gid)].is_seed.sum())
        assert np.isclose(row.sum_kzt_internal, amounts.get(row.cluster_id, 0), rtol=1e-12, atol=1e-6)
        leaders = json.loads(row.top_gids)
        expected = group.sort_values(["betweenness", "pagerank", "gid"],
                                     ascending=[False, False, True]).gid.head(5).tolist()
        assert leaders == expected and len(leaders) > 0
        assert row.hypothesis == cluster_hypothesis(group, row.sum_kzt_internal, roles)
    minima = df.groupby("cluster_id").gid.min().tolist()
    assert minima == sorted(minima), "Неканонические cluster ids"
    assert len(top) == min(TOP_N, len(nodes)) and len(top) >= 20
    assert top.gid.is_unique and top["rank"].tolist() == list(range(1, len(top) + 1))
    expected_top = df.sort_values(["priority_score", "gid"], ascending=[False, True]).head(TOP_N)
    assert top.gid.tolist() == expected_top.gid.tolist()
    assert top.role.tolist() == expected_top.role.tolist()
    assert np.allclose(top.priority_score, expected_top.priority_score, rtol=0, atol=1e-10)
    assert top.why.str.strip().str.len().gt(0).all() and top.why.str.contains(r"\d").all()
    assert (df.loc[df.truncated_by_depth, "score_terminal"] == 0).all()
    assert not (df.loc[df.truncated_by_depth, "role"] == "terminal").any()
    orphan = ~df.gid.isin(set(edges.src) | set(edges.dst))
    assert (df.loc[orphan, "role"] == "peripheral").all()
    assert (df.loc[orphan, "priority_score"] == 0).all()
    assert (df.pass_through_defined == (df.in_kzt > 0)).all()
    assert (df.loc[~df.pass_through_defined, "pass_through"] == -1).all()


def write_outputs(df, clusters, top, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for table, name in ((df, "nodes_roles.csv"), (clusters, "clusters.csv"), (top, "top_nodes.csv")):
        table.to_csv(out_dir / name, index=False, float_format=f"%.{PRECISION}f", lineterminator="\n")
    print(f"Выгрузки записаны в {out_dir}/")
