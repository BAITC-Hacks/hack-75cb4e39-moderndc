#!/usr/bin/env python3
"""
Стартовый код кейса «Граф денег» — HackAlem AI.

Что он делает:
  1. грузит три parquet-файла и проверяет их консистентность;
  2. собирает направленный взвешенный граф;
  3. считает БАЗОВЫЕ метрики узлов (степени, обороты, PageRank);
  4. рассчитывает роли, кластеры и приоритет проверки;
  5. проверяет и пишет три обязательные выгрузки.

Запуск:
    python starter.py --data ./data --out ./out
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from analytics import build_outputs, validate_outputs, write_outputs

ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]


# ---------------------------------------------------------------- загрузка

def load(data_dir: Path):
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def sanity_check(edges, nodes, tx):
    """Проверки, которые стоит пройти до того, как строить модель."""
    print("=" * 64)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 64)
    print(f"  узлов в nodes.parquet : {len(nodes):>6}")
    print(f"  рёбер                 : {len(edges):>6}")
    print(f"  транзакций            : {len(tx):>6}")
    print(f"  seed-клиентов         : {int(nodes.is_seed.sum()):>6}")
    print(f"  оборот, KZT           : {edges.sum_kzt.sum():>14,.0f}")
    print(f"  период                : {tx.date.min().date()} — {tx.date.max().date()}")

    # транзакции должны складываться в рёбра
    agg = tx.groupby(["src", "dst"]).agg(s=("sum_kzt", "sum"), c=("sum_kzt", "size")).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    assert (m._merge == "both").all(), "edges и transactions не сходятся по парам"
    assert np.allclose(m.sum_kzt, m.s, rtol=1e-12, atol=1e-6), "Не сходятся суммы"
    assert (m.n_tx == m.c).all(), "Не сходится число транзакций"
    assert nodes.gid.notna().all() and nodes.gid.is_unique, "gid должны быть уникальны"
    assert all(pd.api.types.is_integer_dtype(s) for s in
               (nodes.gid, edges.src, edges.dst, tx.src, tx.dst)), "Ожидается фактический int gid"
    assert not edges.duplicated(["src", "dst"]).any(), "Повторные пары рёбер"
    assert (set(edges.src) | set(edges.dst)) <= set(nodes.gid), "Неизвестные gid"
    for table in (nodes, edges, tx):
        assert not table.isna().any().any(), "Пропуски во входных данных"
        numeric = table.select_dtypes(include="number").to_numpy()
        assert not np.iscomplexobj(numeric) and np.isfinite(numeric).all()
    assert (edges.sum_kzt > 0).all() and (tx.sum_kzt > 0).all()
    assert (edges.n_tx > 0).all() and (edges.n_tx == edges.n_tx.astype(int)).all()
    assert nodes.depth.between(0, 4).all() and nodes.is_seed.isin([True, False]).all()
    print("  edges == transactions : OK")

    # узлы без единого ребра
    in_edges = set(edges.src) | set(edges.dst)
    orphans = set(nodes.gid) - in_edges
    print(f"\n  ВНИМАНИЕ: {len(orphans)} узлов нет ни в одном ребре "
          f"(из них seed: {len(orphans & set(nodes[nodes.is_seed].gid))})")
    print("  → они всё равно должны попасть в nodes_roles.csv")
    print("=" * 64, "\n")
    return orphans


# ---------------------------------------------------------------- граф

def build_graph(edges, nodes) -> nx.DiGraph:
    """Направленный граф. sum_kzt — вес ребра, n_tx — количество переводов."""
    G = nx.DiGraph()
    # Фактический gid — int64. Числовой порядок сохраняет identity без float.
    G.add_nodes_from(sorted(nodes.gid))
    for r in edges.sort_values(["src", "dst"]).itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G


def basic_features(G: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    """Базовые метрики. Это старт, а не финиш — добавляйте свои."""
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_kzt = dict(G.in_degree(weight="sum_kzt"))
    out_kzt = dict(G.out_degree(weight="sum_kzt"))
    in_tx = dict(G.in_degree(weight="n_tx"))
    out_tx = dict(G.out_degree(weight="n_tx"))
    try:
        pr = nx.pagerank(G, weight="sum_kzt", max_iter=1000, tol=1e-12)
    except nx.PowerIterationFailedConvergence:
        print("FALLBACK: PageRank не сошёлся; равномерный score (без сигнала ранга)")
        pr = dict.fromkeys(G, 1 / len(G) if len(G) else 0.0)

    df = nodes[["gid", "depth", "is_seed"]].sort_values("gid").reset_index(drop=True)
    df["in_deg"] = df.gid.map(in_deg).fillna(0).astype(int)
    df["out_deg"] = df.gid.map(out_deg).fillna(0).astype(int)
    df["in_kzt"] = df.gid.map(in_kzt).fillna(0.0)
    df["out_kzt"] = df.gid.map(out_kzt).fillna(0.0)
    df["in_tx"] = df.gid.map(in_tx).fillna(0).astype(int)
    df["out_tx"] = df.gid.map(out_tx).fillna(0).astype(int)
    df["pagerank"] = df.gid.map(pr).fillna(0.0)

    # доля полученного, которая ушла дальше. Около 1.0 — деньги не задерживаются.
    df["pass_through"] = np.where(df.in_kzt > 0, df.out_kzt / df.in_kzt.replace(0, np.nan), np.nan)

    # ЛОВУШКА КЕЙСА: узел на 4-м колене без исходящих может быть не «стоком»,
    # а просто местом, где закончился обход. Разберитесь с этим.
    df["truncated_by_depth"] = (df.depth == 4) & (df.out_deg == 0)
    return df



# ---------------------------------------------------------------- подсказки

def hints(G: nx.DiGraph, df: pd.DataFrame):
    """Куда смотреть дальше. Ответов здесь нет — только направления."""
    print("\nС ЧЕГО НАЧАТЬ")
    print("-" * 64)
    print(f"  узлов, получающих от 3+ разных плательщиков : {(df.in_deg >= 3).sum()}")
    print(f"  узлов, рассылающих на 10+ получателей       : {(df.out_deg >= 10).sum()}")
    print(f"  узлов и с входом, и с выходом               : {((df.in_deg > 0) & (df.out_deg > 0)).sum()}")
    print(f"  узлов, обрезанных 4-м коленом               : {df.truncated_by_depth.sum()}  <- разберитесь")
    print(f"  слабосвязных компонент                      : {nx.number_weakly_connected_components(G)}")
    print("""
  Вопросы, на которые стоит ответить метриками:
    * чем «деньги пришли и остались» отличается от «пришли и ушли дальше»?
    * что важнее для роли — количество плательщиков или сумма?
    * узел собирает средства от нескольких SEED — это случайность или структура?
    * если убрать узел, сеть распадётся или переживёт?

  Полезное в networkx: pagerank, hits, betweenness_centrality,
  community.louvain_communities, simple_cycles, all_simple_paths.
  Не забудьте: граф НАПРАВЛЕННЫЙ и ВЗВЕШЕННЫЙ.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", help="папка с parquet-файлами")
    ap.add_argument("--out", default="./out", help="куда писать выгрузки")
    a = ap.parse_args()

    edges, nodes, tx = load(Path(a.data))
    sanity_check(edges, nodes, tx)
    G = build_graph(edges, nodes)
    df = basic_features(G, nodes)
    outputs = build_outputs(G, df, tx, ROLES)
    validate_outputs(*outputs, nodes, edges, ROLES)
    write_outputs(*outputs, Path(a.out))
    # Проверяется и сериализованный результат, с gid строго int64.
    restored = [pd.read_csv(Path(a.out) / name) for name in
                ("nodes_roles.csv", "clusters.csv", "top_nodes.csv")]
    validate_outputs(*restored, nodes, edges, ROLES)
    print("PASS: входы, граф и все три CSV проверены")


if __name__ == "__main__":
    main()
