"""Read-only retrieval contracts and Streamlit smoke checks; no scoring tests."""

import json
from decimal import Decimal
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

    def assert_table(self, container, frame):
        # Independent display expectations; still compare every cell and row in order.
        shown = frame.copy(deep=True)
        labels = {"rank": "Rank", "gid": "GID", "role": "Role", "role_score": "Role score",
                  "priority_score": "Review priority", "why": "Key signals", "evidence": "Evidence",
                  "cluster_id": "Cluster ID", "n_nodes": "Members", "n_seed": "Seed nodes",
                  "sum_kzt_internal": "Internal flow", "top_gids": "Top GIDs",
                  "hypothesis": "Structural hypothesis", "direction": "Direction",
                  "neighbor_gid": "Neighbor GID", "sum_kzt": "Amount", "n_tx": "Transactions"}
        for key in shown:
            if key == "top_gids":
                shown[key] = shown[key].map(lambda value: "\n".join(str(gid) for gid in json.loads(value)))
            elif key in ("why", "hypothesis"):
                def description(raw):
                    parts = []
                    names = {"bet_pct": "Betweenness percentile", "pr_pct": "PageRank percentile",
                             "volume_pct": "Volume percentile", "tx_pct": "Transaction count percentile",
                             "prev_day": "Prior-day activity signal", "dominant_share": "Dominant share",
                             "category": "Orientation", "dominant_role": "Dominant role",
                             "truncated": "Truncated nodes"}
                    for token in raw.split("; "):
                        field, _, value = token.partition("=")
                        if token.startswith("roles: "):
                            text = "Role counts: " + token[7:].replace("=", ": ")
                        elif field in ("bet_pct", "pr_pct", "volume_pct", "tx_pct", "prev_day", "dominant_share"):
                            text = f"{names[field]}: {Decimal(value) * 100:.2f}%"
                        elif field == "temporal":
                            text = f"Temporal priority component: {Decimal(value):.4f}"
                        elif field == "internal":
                            text = "Internal flow: ₸" + format(Decimal(value.removesuffix(" KZT")).normalize(), ",f")
                        elif field in names:
                            text = f"{names[field]}: {value}"
                        else:
                            text = "Structural hypothesis only" if token == "structural hypothesis only" else token
                        parts.append(text)
                    return " · ".join(parts)
                shown[key] = shown[key].map(description)
            elif key in ("sum_kzt", "sum_kzt_internal"):
                shown[key] = shown[key].map(lambda v: "₸" + format(Decimal(str(v)).normalize(), ",f"))
            elif key in ("role_score", "priority_score"):
                shown[key] = shown[key].map(lambda v: f"{Decimal(str(v)):.4f}")
            elif key == "direction":
                shown[key] = shown[key].map({"incoming": "Incoming", "outgoing": "Outgoing"})
        if "evidence" in shown:
            # Independent expectations from the stored serialization, not app's formatter.
            from html import unescape
            import re

            def evidence_text(raw):
                signals = []
                for token in raw.split(";"):
                    key, value = token.strip().split("=", 1)
                    text = token.strip()
                    if key in ("in_deg", "out_deg", "depth"):
                        text = {"in_deg": f"{value} incoming", "out_deg": f"{value} outgoing",
                                "depth": f"Depth {value}"}[key]
                    elif key in ("in", "out"):
                        amount = Decimal(value.removesuffix(" KZT"))
                        money = format(amount.normalize(), ",f")
                        for scale, suffix in ((10**9, "B"), (10**6, "M"), (10**3, "K")):
                            if amount >= scale:
                                money = f"{amount / scale:.2f}".rstrip("0").rstrip(".") + suffix
                                break
                        text = f"₸{money} {key}"
                    elif key == "tx":
                        incoming, outgoing = value.split("/")
                        text = f"{incoming} incoming tx · {outgoing} outgoing tx"
                    elif key in ("bet_pct", "pr_pct", "prev_day"):
                        label = {"bet_pct": "Betweenness", "pr_pct": "PageRank",
                                 "prev_day": "Prior-day activity signal"}[key]
                        text = f"{label} {Decimal(value) * 100:.2f}%"
                    elif key in ("truncated", "is_seed"):
                        text = {"truncated": {"0": "Not truncated", "1": "Truncated"},
                                "is_seed": {"0": "Non-seed", "1": "Seed"}}[key][value]
                    elif key == "role_score":
                        text = f"Role score {Decimal(value):.4f}"
                    elif key == "observed_edges" and value == "0":
                        text = "0 observed edges"
                    elif key == "out/in" and value == "unavailable":
                        text = "Out/in ratio unavailable"
                    signals.append(text)
                return " · ".join(signals)

            def visible_rows(html):
                # AppTest exposes these escaped tables as HTML; compare visible cells,
                # ignoring styling, nesting attributes and whitespace, not row order.
                return [[" ".join(unescape(cell).split()) for cell in
                         re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", row, re.S)]
                        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, re.S)]

            shown["evidence"] = shown.evidence.map(evidence_text)
            expected = shown.rename(columns=labels).to_html(index=False, escape=True)
            self.assertIn(visible_rows(expected),
                          [visible_rows(item.proto.body) for item in container.get("html")])
            return
        expected = shown.rename(columns=labels).to_html(index=False, escape=True, border=0, classes="inspection-table",
                                                       formatters={"Top GIDs": lambda value: value})
        self.assertIn(expected, [item.proto.body for item in container.get("html")])

    def collapsed(self, at, label):
        block = next(item for item in at.expander if item.label == label)
        self.assertFalse(block.proto.expanded)
        return block

    def test_dashboard_previews(self):
        at = self.app()
        self.assertFalse(at.exception)
        top, clusters = self.data["top"], self.data["clusters"]
        self.assert_table(at, top.head(10))
        self.assert_table(self.collapsed(at, "Show all priority nodes"), top)
        expected = clusters.sort_values(["n_nodes", "cluster_id"], ascending=[False, True]).head(10)
        self.assert_table(at, expected[["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "hypothesis"]])
        self.assert_table(self.collapsed(at, "Show all clusters"), clusters)
        self.assertIn(f"Showing top {min(10, len(top))} of {len(top)} priority nodes.", [v.value for v in at.caption])
        self.assertIn(f"Showing {len(expected)} of {len(clusters)} clusters.", [v.value for v in at.caption])

    def test_node_relationship_disclosure(self):
        agent = Analyst(self.data)
        gid = self.data["top"].iloc[0].gid
        relationships = agent.get_neighbors(gid)
        self.assertGreater(len(relationships), 30)
        at = self.app()
        at.radio(key="page").set_value("Node Inspector").run()
        at.selectbox(key="node_gid").set_value(gid).run()
        self.assertFalse(at.exception)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Incoming relationships"], str(sum(r["direction"] == "incoming" for r in relationships)))
        self.assertEqual(metrics["Outgoing relationships"], str(sum(r["direction"] == "outgoing" for r in relationships)))
        self.assertEqual(metrics["Observed incident edges"], str(len(relationships)))
        self.assertIn(f"Showing 30 of {len(relationships)} observed incident edges in the graph. "
                      f"All {len(relationships)} observed relationships remain available below.", [v.value for v in at.caption])
        frame = pd.DataFrame(relationships)
        self.assert_table(at, frame.head(15))
        self.assert_table(self.collapsed(at, f"Show all {len(frame)} observed relationships"), frame)
        self.assertEqual(agent.get_neighbors(gid), relationships)
        small = self.nodes[(self.nodes.in_deg + self.nodes.out_deg).between(1, 15)].iloc[0].gid
        at.selectbox(key="node_gid").set_value(small).run()
        self.assertFalse(at.exception)
        self.assertFalse(any("observed incident edges in the graph" in v.value for v in at.caption))
        self.assert_table(at, pd.DataFrame(agent.get_neighbors(small)))
        self.assertFalse(any("observed relationships" in v.label for v in at.expander))
        isolated = self.nodes[(self.nodes.in_deg + self.nodes.out_deg) == 0].iloc[0].gid
        at.selectbox(key="node_gid").set_value(isolated).run()
        metrics = {m.label: m.value for m in at.metric}
        for label in ("Incoming relationships", "Outgoing relationships", "Observed incident edges"):
            self.assertEqual(metrics[label], "0")

    def test_cluster_summary_and_disclosure(self):
        from app import parse_hypothesis
        cid = int(self.data["clusters"].sort_values("n_nodes", ascending=False).iloc[0].cluster_id)
        detail = Analyst(self.data).inspect_cluster(cid)
        hypothesis = detail["cluster"]["hypothesis"]
        stored = dict(part.split("=", 1) for part in hypothesis.split("; ") if "=" in part)
        expected = {"Structural category": stored["category"], "Dominant role": stored["dominant_role"],
                    "Dominant share": stored["dominant_share"], "Internal KZT": stored["internal"],
                    "Truncated nodes": stored["truncated"]}
        self.assertEqual(parse_hypothesis(hypothesis), expected)
        self.assertEqual(parse_hypothesis("unknown layout"), {})
        self.assertEqual(parse_hypothesis(None), {})
        self.assertEqual(parse_hypothesis("category=mixed; category=collection-oriented; truncated=2"), {"Truncated nodes": "2"})
        self.assertEqual(parse_hypothesis("dominant_share=NaN; internal=unknown; truncated=4"), {"Truncated nodes": "4"})
        at = self.app()
        at.radio(key="page").set_value("Cluster Inspector").run()
        at.selectbox(key="cluster_id").set_value(cid).run()
        self.assertFalse(at.exception)
        displayed = dict(expected)
        displayed["Dominant share"] = f"{Decimal(stored['dominant_share']) * 100:.2f}%"
        amount = Decimal(str(detail["cluster"]["sum_kzt_internal"]))
        for scale, suffix in ((10**9, "B"), (10**6, "M"), (10**3, "K"), (1, "")):
            if amount >= scale:
                displayed["Internal KZT"] = "₸" + f"{amount / scale:.2f}".rstrip("0").rstrip(".") + suffix
                break
        self.assertEqual({m.label: m.value for m in at.metric}, displayed)
        full = self.collapsed(at, "Full stored structural hypothesis")
        self.assertIn(hypothesis, [item.value for item in full.markdown])
        members = pd.DataFrame(detail["members"])[["gid", "role", "priority_score", "evidence"]]
        self.assert_table(at, members.head(15))
        self.assert_table(self.collapsed(at, f"Show all {len(members)} members"), members)
        # Inject only presentation text; membership/action semantics remain untouched.
        unexpected = "Unexpected hypothesis format; original text must remain intact."
        changed = {**detail, "cluster": {**detail["cluster"], "hypothesis": unexpected}}
        with patch.object(Analyst, "inspect_cluster", return_value=changed):
            at.run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.metric), 0)
        self.assertIn(unexpected, [m.value for m in self.collapsed(at, "Full stored structural hypothesis").markdown])
        small = int(self.data["clusters"].loc[self.data["clusters"].n_nodes <= 15].iloc[0].cluster_id)
        at.selectbox(key="cluster_id").set_value(small).run()
        self.assertFalse(at.exception)
        frame = pd.DataFrame(Analyst(self.data).inspect_cluster(small)["members"])[["gid", "role", "priority_score", "evidence"]]
        self.assert_table(at, frame)
        self.assertFalse(any(v.label.endswith(" members") for v in at.expander))

    def test_agent_human_results(self):
        agent = Analyst(self.data)
        normal = self.nodes.iloc[0]
        actions = [("inspect_node", normal.gid), ("inspect_cluster", str(normal.cluster_id)),
                   ("get_top_priority", "10"), ("get_neighbors", normal.gid),
                   ("explain_node", normal.gid),
                   ("explain_node", self.nodes[self.nodes.is_seed].iloc[0].gid),
                   ("explain_node", self.nodes[self.nodes.truncated_by_depth].iloc[0].gid)]
        at = self.app()
        at.radio(key="page").set_value("Agentic Analyst").run()
        for action, value in actions:
            with self.subTest(action=action, value=value):
                at.selectbox[0].set_value(action)
                at.text_input[0].set_value(value)
                at.button[0].click().run()
                self.assertFalse(at.exception)
                expected = agent.dispatch(action, value)
                result = expected["data"]
                raw = self.collapsed(at, "Raw structured result")
                self.assertEqual(json.loads(raw.json[0].value), result)
                self.assertEqual(at.code[0].value, "\n".join(expected["trace"]))
                if action == "explain_node":
                    self.assertIn(result["explanation"], [v.value for v in at.markdown])
                    self.assertEqual([v.value for v in at.warning], result["limitations"])
                elif action == "inspect_node":
                    self.assert_table(at, pd.DataFrame([result])[["gid", "role", "role_score", "priority_score", "evidence"]])
                elif action == "inspect_cluster":
                    self.assert_table(at, pd.DataFrame([result["cluster"]])[["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal"]])
                else:
                    self.assert_table(at, pd.DataFrame(result).head(15) if action == "get_neighbors" else pd.DataFrame(result))

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
        self.assertIn("Technical evidence", [v.value for v in at.caption])
        summary_text = "\n".join(displayed)
        self.assertIn(f"Incoming relationships: {normal.in_deg}", summary_text)
        self.assertIn(f"Outgoing transactions: {normal.out_tx}", summary_text)
        self.assertIn(f"Betweenness percentile: {Decimal(str(normal.betweenness_pct)) * 100:.2f}%", summary_text)
        from app import display_frame, format_kzt, format_percent, format_score
        original = self.data["top"].copy(deep=True)
        display_frame(self.data["top"])
        pd.testing.assert_frame_equal(self.data["top"], original)
        self.assertEqual(format_kzt("18838290.01"), "₸18,838,290.01")
        self.assertEqual(format_kzt("3848440", compact=True), "₸3.85M")
        self.assertEqual(format_kzt("0.001"), "₸0.001")
        self.assertEqual(format_percent("0.5724637681"), "57.25%")
        self.assertEqual(format_score("0.975123"), "0.9751")
        for missing in (None, float("nan"), float("inf")):
            for formatter in (format_kzt, format_percent, format_score):
                self.assertEqual(formatter(missing), "Unavailable")
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
        all_expected = expected.copy()
        expected = expected.assign(_src=expected.src.map(int), _dst=expected.dst.map(int)).sort_values(
            ["sum_kzt", "n_tx", "_src", "_dst"], ascending=[False, False, True, True]).head(30)
        actual = fig.layout.meta["observed_edges"]
        self.assertEqual(actual, expected[["src", "dst", "sum_kzt", "n_tx"]].to_dict("records"))
        self.assertLessEqual(len(actual), 30)
        self.assertTrue({(r["src"], r["dst"]) for r in actual} <= set(zip(all_expected.src, all_expected.dst)))
        self.assertEqual({(r["src"], r["dst"], r["sum_kzt"], r["n_tx"]) for r in actual},
                         set(expected[["src", "dst", "sum_kzt", "n_tx"]].itertuples(index=False, name=None)))
        self.assertEqual(len(fig.layout.annotations), len(expected))
        self.assertEqual(set(fig.layout.meta["node_gids"]), set(expected.src) | set(expected.dst) | {gid})
        for trace, arrow, edge in zip(fig.data[:-1], fig.layout.annotations, actual):
            self.assertEqual(trace.text[0],
                             f"Source GID: {edge['src']}<br>Destination GID: {edge['dst']}"
                             f"<br>Amount: ₸{format(Decimal(str(edge['sum_kzt'])).normalize(), ',f')}"
                             f"<br>Transactions: {edge['n_tx']}")
            self.assertTrue(arrow.showarrow)
            self.assertEqual((arrow.ax, arrow.ay), (trace.x[-6], trace.y[-6]))
            self.assertEqual((arrow.x, arrow.y), (trace.x[-3], trace.y[-3]))
        for node_gid, tooltip in zip(fig.layout.meta["node_gids"], fig.data[-1].text):
            node = self.nodes.loc[self.nodes.gid == node_gid].iloc[0]
            self.assertEqual(tooltip, f"GID: {node_gid}<br>Role: {node.role}"
                             f"<br>Review priority: {Decimal(str(node.priority_score)):.4f}")
        isolated = self.nodes[(self.nodes.in_deg + self.nodes.out_deg) == 0].iloc[0].gid
        self.assertIsNone(local_graph(self.data, isolated))
        small = self.nodes[(self.nodes.in_deg + self.nodes.out_deg).between(1, 30)].iloc[0].gid
        actual_small = local_graph(self.data, small).layout.meta["observed_edges"]
        expected_small = edges[(edges.src == small) | (edges.dst == small)]
        self.assertEqual({(r["src"], r["dst"]) for r in actual_small}, set(zip(expected_small.src, expected_small.dst)))

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
