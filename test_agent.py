"""Read-only retrieval contracts and Streamlit smoke checks; no scoring tests."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from agent import (Analyst, DIAGNOSTICS, MISSING_OUTPUT, SEED_WARNING, TRUNCATION_WARNING,
                   load_data, role_counts, summary)

ROOT = Path(__file__).resolve().parent


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data(ROOT)
        cls.agent = Analyst(cls.data)
        cls.raw_nodes = pd.read_csv(ROOT / "out/nodes_roles.csv", dtype={"gid": "string"})
        cls.raw_edges = pd.read_parquet(ROOT / "data/edges.parquet")
        cls.gid = cls.raw_nodes.iloc[0].gid
        cls.isolate = cls.raw_nodes.loc[(cls.raw_nodes.in_deg + cls.raw_nodes.out_deg) == 0].iloc[0].gid
        cls.cluster = int(cls.raw_nodes.iloc[0].cluster_id)

    def test_inspect_node_valid(self):
        self.assertEqual(self.agent.inspect_node(self.gid), self.raw_nodes.iloc[0].to_dict())
        self.assertEqual(self.agent.inspect_node(int(self.gid)), self.agent.inspect_node(self.gid))

    def test_invalid_inputs(self):
        for action, values in {"inspect_node": ["not-a-gid", "0", float(self.gid), True, "1e17"],
                               "inspect_cluster": ["-1", "999999", "abc", 0.0],
                               "get_top_priority": [0, -1, 31, 1.5]}.items():
            for value in values:
                with self.subTest(action=action, value=value):
                    result = self.agent.dispatch(action, value)
                    self.assertFalse(result["ok"])
                    self.assertTrue(result["error"])
                    self.assertFalse(any(line.startswith("RESULT:") for line in result["trace"]))
        self.assertFalse(self.agent.dispatch("unknown", self.gid)["ok"])

    def test_inspect_cluster_valid(self):
        result = self.agent.inspect_cluster(self.cluster)
        raw = pd.read_csv(ROOT / "out/clusters.csv")
        self.assertEqual(result["cluster"], raw[raw.cluster_id == self.cluster].iloc[0].to_dict())
        members = self.raw_nodes[self.raw_nodes.cluster_id == self.cluster]
        expected = members.assign(_gid=members.gid.map(int)).sort_values(
            ["priority_score", "_gid"], ascending=[False, True]).drop(columns="_gid")
        self.assertEqual(result["members"], expected.to_dict("records"))
        self.assertEqual({k: v for k, v in result["role_distribution"].items() if v}, members.role.value_counts().to_dict())

    def test_top_priority(self):
        raw = pd.read_csv(ROOT / "out/top_nodes.csv", dtype={"gid": "string"})
        self.assertEqual(self.agent.get_top_priority(10), raw.head(10).to_dict("records"))
        self.assertEqual(self.agent.get_top_priority(len(raw)), raw.to_dict("records"))

    def test_neighbors_with_edges(self):
        # Use a node with both directions to catch accidental reversal.
        gid = str(self.raw_nodes.loc[(self.raw_nodes.in_deg > 0) & (self.raw_nodes.out_deg > 0)].iloc[0].gid)
        actual = self.agent.get_neighbors(gid)
        expected = []
        for direction, endpoint, neighbor in (("incoming", "dst", "src"), ("outgoing", "src", "dst")):
            for row in self.raw_edges[self.raw_edges[endpoint] == int(gid)].sort_values(neighbor).to_dict("records"):
                expected.append({"direction": direction, "neighbor_gid": str(row[neighbor]),
                                 "sum_kzt": row["sum_kzt"], "n_tx": row["n_tx"]})
        self.assertEqual(actual, expected)

    def test_neighbors_isolated(self):
        self.assertEqual(self.agent.get_neighbors(self.isolate), [])
        self.assertEqual(self.agent.inspect_node(self.isolate)["role"], "peripheral")

    def test_explain_normal(self):
        row = self.raw_nodes[~self.raw_nodes.is_seed & ~self.raw_nodes.truncated_by_depth].iloc[0]
        result = self.agent.explain_node(row.gid)
        for key in ("role", "role_score", "priority_score", "evidence"):
            self.assertEqual(result[key], row[key])
        self.assertEqual(result["diagnostics"], {key: row[key] for key in DIAGNOSTICS if key in row})
        self.assertEqual(result["limitations"], [])
        self.assertIn("explanation", result)
        text = result["explanation"]
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())
        expected = (f"gid {row.gid} is observed as {row.role} (role_score={row.role_score:.4f}). "
                    f"Evidence: {row.evidence}. Review priority={row.priority_score:.4f}. "
                    "Priority is a review priority, not a probability or proof of wrongdoing.")
        self.assertEqual(text, expected)
        self.assertIn(row.evidence, text)
        self.assertEqual(text.encode("utf-8"), self.agent.explain_node(row.gid)["explanation"].encode("utf-8"))
        self.assertEqual(self.agent.dispatch("explain_node", row.gid)["data"]["explanation"], text)
        self.assertNotIn(SEED_WARNING, text)
        self.assertNotIn(TRUNCATION_WARNING, text)

    def test_explain_seed(self):
        row = self.raw_nodes[self.raw_nodes.is_seed].iloc[0]
        self.assertIn(SEED_WARNING, self.agent.explain_node(row.gid)["limitations"])

    def test_explain_truncated_and_both(self):
        row = self.raw_nodes[self.raw_nodes.truncated_by_depth].iloc[0]
        self.assertIn(TRUNCATION_WARNING, self.agent.explain_node(row.gid)["limitations"])
        controlled = {key: value.copy() for key, value in self.data.items()}
        controlled["nodes"].loc[controlled["nodes"].gid == row.gid, "is_seed"] = True
        self.assertEqual(Analyst(controlled).explain_node(row.gid)["limitations"], [SEED_WARNING, TRUNCATION_WARNING])

    def test_deterministic_dispatch(self):
        for action, value in (("inspect_node", self.gid), ("inspect_cluster", self.cluster),
                              ("get_top_priority", 10), ("get_neighbors", self.gid), ("explain_node", self.gid)):
            first = self.agent.dispatch(action, value)
            self.assertTrue(first["ok"])
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(self.agent.dispatch(action, value), sort_keys=True))
            self.assertEqual(first["data"], getattr(self.agent, action)(value))
            self.assertIn(f"DISPATCH: {action}", first["trace"])

    def test_summary_sources(self):
        self.assertEqual(summary(self.data), {"Nodes": 2248, "Edges": 3119, "Transactions": 4840,
                                             "Clusters": 88, "Seed nodes": 81, "Truncated depth-4 nodes": 444})
        self.assertEqual(summary(self.data)["Transactions"], int(self.raw_edges.n_tx.sum()))
        self.assertEqual(role_counts(self.data["nodes"]), self.raw_nodes.role.value_counts().to_dict())

    def test_missing_malformed_and_optional_fields(self):
        with patch("agent.Path.is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "Analytics outputs are missing"):
                load_data(ROOT)
        original = pd.read_csv
        def reader(path, **kwargs):
            frame = original(path, **kwargs)
            if Path(path).name == "nodes_roles.csv":
                return frame.drop(columns=[column for column in DIAGNOSTICS if column in frame])
            return frame
        with patch("agent.pd.read_csv", side_effect=reader):
            minimal = load_data(ROOT)
        self.assertEqual(Analyst(minimal).explain_node(self.gid)["diagnostics"], {})
        def malformed(path, **kwargs):
            frame = original(path, **kwargs)
            return frame.drop(columns="role") if Path(path).name == "nodes_roles.csv" else frame
        with patch("agent.pd.read_csv", side_effect=malformed):
            with self.assertRaisesRegex(ValueError, "missing columns: role"):
                load_data(ROOT)


class UISmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data(ROOT)
        cls.nodes = cls.data["nodes"]

    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def test_dashboard_node_cluster_agent_paths(self):
        at = self.app()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual({item.label: item.value for item in at.metric},
                         {key: f"{value:,}" for key, value in summary(self.data).items()})
        at.radio(key="page").set_value("Node Inspector").run()
        self.assertEqual(len(at.exception), 0)
        normal = self.nodes[~self.nodes.is_seed & ~self.nodes.truncated_by_depth].iloc[0]
        at.selectbox(key="node_gid").set_value(normal.gid).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.metric[0].value, normal.role)
        explanation = Analyst(self.data).explain_node(normal.gid)["explanation"]
        displayed = [item.value for item in at.markdown]
        self.assertIn(explanation, displayed)
        self.assertIn(normal.gid, explanation)
        self.assertIn(normal.role, explanation)
        self.assertEqual(sum(text.count(normal.evidence) for text in displayed), 1)
        self.assertNotIn(normal.evidence, displayed)
        self.assertEqual(len(at.get("plotly_chart")), 1)
        for mask, warning in ((self.nodes.is_seed, SEED_WARNING), (self.nodes.truncated_by_depth, TRUNCATION_WARNING)):
            gid = self.nodes[mask].iloc[0].gid
            at.selectbox(key="node_gid").set_value(gid).run()
            self.assertEqual(len(at.exception), 0)
            self.assertIn(warning, [item.value for item in at.warning])
        isolated = self.nodes[(self.nodes.in_deg + self.nodes.out_deg) == 0].iloc[0].gid
        at.selectbox(key="node_gid").set_value(isolated).run()
        self.assertEqual(len(at.exception), 0)
        self.assertIn("No observed incoming or outgoing edges.", [item.value for item in at.info])
        at.text_input(key="exact_gid").set_value("invalid").run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(at.error)
        at.radio(key="page").set_value("Cluster Inspector").run()
        at.selectbox(key="cluster_id").set_value(int(normal.cluster_id)).run()
        self.assertEqual(len(at.exception), 0)
        at.text_input(key="exact_cluster").set_value("999999").run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(at.error)
        at.radio(key="page").set_value("Agentic Analyst").run()
        at.text_input[0].set_value(normal.gid)
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(at.success)
        self.assertIn("DISPATCH: inspect_node", at.code[0].value)

    def test_graph_observed_direction_and_tooltips(self):
        from app import local_graph
        gid = self.data["top"].iloc[0].gid
        fig = local_graph(self.data, gid)
        edges = self.data["edges"]
        expected = edges[(edges.src == gid) | (edges.dst == gid)]
        actual = fig.layout.meta["observed_edges"]
        self.assertEqual({(r["src"], r["dst"], r["sum_kzt"], r["n_tx"]) for r in actual},
                         set(expected[["src", "dst", "sum_kzt", "n_tx"]].itertuples(index=False, name=None)))
        self.assertEqual(len(fig.layout.annotations), len(expected))
        self.assertEqual(set(fig.layout.meta["node_gids"]), set(expected.src) | set(expected.dst) | {gid})
        for trace, arrow, edge in zip(fig.data[:-1], fig.layout.annotations, actual):
            self.assertIn(f"{edge['src']} → {edge['dst']}", trace.text[0])
            self.assertIn("sum_kzt=", trace.text[0])
            self.assertIn("n_tx=", trace.text[0])
            self.assertTrue(arrow.showarrow)
            self.assertEqual((arrow.ax, arrow.ay), (trace.x[-6], trace.y[-6]))
            self.assertEqual((arrow.x, arrow.y), (trace.x[-3], trace.y[-3]))
        for tooltip in fig.data[-1].text:
            self.assertIn("gid=", tooltip)
            self.assertIn("role=", tooltip)
            self.assertIn("priority_score=", tooltip)
        isolated = self.nodes[(self.nodes.in_deg + self.nodes.out_deg) == 0].iloc[0].gid
        self.assertIsNone(local_graph(self.data, isolated))

    def test_missing_and_malformed_ui(self):
        import streamlit as st
        for message in (MISSING_OUTPUT, "Cannot load presentation data: missing columns: role."):
            st.cache_data.clear()
            with patch("agent.load_data", side_effect=ValueError(message)):
                at = self.app()
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(at.error[0].value, message)
        st.cache_data.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
