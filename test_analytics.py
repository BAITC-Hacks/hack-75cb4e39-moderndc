"""Проверки смысловых ограничений кейса и полного pipeline на реальных данных."""

import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd

import analytics
import starter


class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.edges, cls.nodes, cls.tx = starter.load(Path(__file__).parent / "data")
        cls.G = starter.build_graph(cls.edges, cls.nodes)
        cls.base = starter.basic_features(cls.G, cls.nodes)
        cls.outputs = analytics.build_outputs(cls.G, cls.base, cls.tx, starter.ROLES)

    def test_full_validation_and_serialization(self):
        with contextlib.redirect_stdout(io.StringIO()):
            starter.sanity_check(self.edges, self.nodes, self.tx)
        analytics.validate_outputs(*self.outputs, self.nodes, self.edges, starter.ROLES)
        with tempfile.TemporaryDirectory() as folder:
            analytics.write_outputs(*self.outputs, Path(folder))
            restored = [pd.read_csv(Path(folder) / name) for name in
                        ("nodes_roles.csv", "clusters.csv", "top_nodes.csv")]
            analytics.validate_outputs(*restored, self.nodes, self.edges, starter.ROLES)

    def test_directed_graph_and_amounts(self):
        self.assertTrue(self.G.is_directed())
        self.assertEqual(set(self.G), set(self.nodes.gid))
        self.assertEqual(self.G.number_of_edges(), len(self.edges))
        df = self.outputs[0].set_index("gid")
        for direction, endpoint in (("in", "dst"), ("out", "src")):
            grouped = self.edges.groupby(endpoint)
            for suffix, expected in (("deg", grouped.size()), ("kzt", grouped.sum_kzt.sum()),
                                     ("tx", grouped.n_tx.sum())):
                np.testing.assert_allclose(df[direction + "_" + suffix],
                                           expected.reindex(df.index, fill_value=0))
        with patch.object(analytics.nx, "betweenness_centrality", wraps=nx.betweenness_centrality) as metric:
            # Миниатюрный граф: проверяем именно semantics weight=None.
            nodes = pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 1, 2], "is_seed": [True, False, False]})
            edges = pd.DataFrame({"src": [1, 2], "dst": [2, 3], "sum_kzt": [10., 100.],
                                  "n_tx": [1, 1], "depth": [1, 2]})
            tx = pd.DataFrame({"src": [1, 2], "dst": [2, 3], "sum_kzt": [10., 100.],
                               "date": pd.to_datetime(["2026-07-01", "2026-07-02"])})
            graph = starter.build_graph(edges, nodes)
            analytics.build_outputs(graph, starter.basic_features(graph, nodes), tx, starter.ROLES)
            self.assertEqual(metric.call_count, 1)
            self.assertIsNone(metric.call_args.kwargs["weight"])

    def test_temporal_day_resolution(self):
        df = self.outputs[0]
        incoming = {gid: set(group.date) for gid, group in self.tx.groupby("dst")}
        outgoing = {gid: group.date.tolist() for gid, group in self.tx.groupby("src")}
        for r in df.itertuples(index=False):
            dates = incoming.get(r.gid, set())
            expected = sum(date - pd.Timedelta(days=1) in dates for date in outgoing.get(r.gid, []))
            self.assertEqual(r.out_after_prior_day_in_tx, expected)
        tx = pd.DataFrame({"src": [1, 2, 2, 2], "dst": [2, 3, 3, 3],
                           "date": pd.to_datetime(["2026-07-02", "2026-07-01", "2026-07-02", "2026-07-03"])})
        tiny = pd.DataFrame({"gid": [2, 4], "out_tx": [3, 0]})
        analytics.temporal_features(tiny, tx)
        self.assertEqual(tiny.out_after_prior_day_in_tx.tolist(), [1, 0])
        self.assertEqual(tiny.active_days.tolist(), [3, 0])
        self.assertAlmostEqual(tiny.prior_day_in_fraction.iloc[0], 1 / 3)

    def test_seed_and_undefined_ratio_do_not_drive_roles(self):
        df = self.outputs[0].copy()
        # Perturbation: менять ratio для seed и undefined не должно менять score любой роли.
        mask = df.is_seed | ~df.pass_through_defined
        df.loc[mask, "pass_through"] = 1e9
        analytics.assign_roles(df, starter.ROLES)
        columns = ["score_" + role for role in starter.ROLES] + ["role", "role_score"]
        pd.testing.assert_frame_equal(df[columns], self.outputs[0][columns])

    def test_truncation_isolates_and_ties(self):
        df = self.outputs[0]
        self.assertTrue((df.loc[df.truncated_by_depth, "score_terminal"] == 0).all())
        orphans = df[(df.in_deg + df.out_deg) == 0]
        self.assertEqual(len(orphans), 19)
        self.assertTrue(orphans.role.eq("peripheral").all())
        self.assertTrue(orphans.score_peripheral.eq(1).all())
        self.assertTrue(orphans.priority_score.eq(0).all())
        # Нетрuncated leaf разрешает terminal, тот же leaf на depth=4 — нет.
        leaf = df[(df.in_deg == 1) & (df.out_deg == 0)].head(1).copy()
        pair = pd.concat([leaf, leaf], ignore_index=True)
        pair["truncated_by_depth"] = [False, True]
        pair["depth"] = [3, 4]
        pair["is_seed"] = False
        # Константные признаки дают нулевой ранг; вводим фон фактических узлов.
        sample = pd.concat([df, pair], ignore_index=True)
        analytics.assign_roles(sample, starter.ROLES)
        self.assertGreater(sample.iloc[-2].score_terminal, 0)
        self.assertEqual(sample.iloc[-1].score_terminal, 0)
        self.assertEqual(starter.ROLES, ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"])

    def test_normalization_and_pagerank_fallback(self):
        for values in ([0, 0], [7, 7], [0, 1, 1, 10]):
            result = analytics.percentile(values)
            self.assertTrue(result.between(0, 1).all())
            if len(set(values)) == 1:
                self.assertTrue(result.eq(0).all())
        for values in ([np.nan], [np.inf], [-1]):
            with self.assertRaises(AssertionError):
                analytics.percentile(values)
        with patch.object(starter.nx, "pagerank", side_effect=nx.PowerIterationFailedConvergence(1000)):
            fallback = starter.basic_features(self.G, self.nodes)
        self.assertEqual(fallback.pagerank.nunique(), 1)
        self.assertTrue(analytics.percentile(fallback.pagerank).eq(0).all())

    def test_sanity_rejects_inconsistent_counts_and_money(self):
        for column in ("sum_kzt", "n_tx"):
            broken = self.edges.copy()
            broken.loc[0, column] += 1
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
                starter.sanity_check(broken, self.nodes, self.tx)

    def test_repeat_and_input_order_determinism(self):
        # Перестановка входов — более сильная проверка, чем просто повторный запуск.
        nodes = self.nodes.sample(frac=1, random_state=17)
        edges = self.edges.sample(frac=1, random_state=23)
        tx = self.tx.sample(frac=1, random_state=31)
        graph = starter.build_graph(edges, nodes)
        repeat = analytics.build_outputs(graph, starter.basic_features(graph, nodes), tx, starter.ROLES)
        for original, repeated in zip(self.outputs, repeat):
            pd.testing.assert_frame_equal(original, repeated, check_exact=True)
        for original, repeated in zip(self.outputs, repeat):
            first = original.to_csv(index=False, float_format="%.10f", lineterminator="\n").encode()
            second = repeated.to_csv(index=False, float_format="%.10f", lineterminator="\n").encode()
            self.assertEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
