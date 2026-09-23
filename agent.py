"""Read-only actions over Stage 1 files. No analytical model is run here."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROLES = ("consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral")
FILES = {"nodes": "out/nodes_roles.csv", "clusters": "out/clusters.csv",
         "top": "out/top_nodes.csv", "edges": "data/edges.parquet"}
REQUIRED = {
    "nodes": ("gid", "role", "role_score", "priority_score", "cluster_id", "depth",
              "is_seed", "truncated_by_depth", "evidence"),
    "clusters": ("cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"),
    "top": ("rank", "gid", "role", "priority_score", "why"),
    "edges": ("src", "dst", "sum_kzt", "n_tx"),
}
DIAGNOSTICS = ("in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx", "pass_through",
               "pass_through_defined", "pagerank", "betweenness", "pagerank_pct", "betweenness_pct",
               "active_days", "activity_span_days", "prior_day_in_fraction")
SEED_WARNING = ("Observed incoming flow may be incomplete because the network "
                "was expanded outward from seed nodes.")
TRUNCATION_WARNING = ("Downstream activity is censored at collection depth 4. "
                      "No observed outgoing edge is not proof that funds stopped here.")
PRIORITY_NOTE = "Priority score is a review priority, not a probability or proof of wrongdoing."
MISSING_OUTPUT = "Analytics outputs are missing. Run:\npython starter.py --data data --out out"


def integer_input(value, name):
    # Never accept float, bool, scientific notation, or a rounded browser number.
    if isinstance(value, (bool, float, np.floating)) or not re.fullmatch(r"[0-9]+", str(value).strip()):
        raise ValueError(f"Invalid {name}: enter an exact non-negative integer.")
    return int(str(value).strip())


def load_data(root: Path):
    root = Path(root)
    if any(not (root / FILES[key]).is_file() for key in ("nodes", "clusters", "top")):
        raise ValueError(MISSING_OUTPUT)
    if not (root / FILES["edges"]).is_file():
        raise ValueError("Observed relationships are missing: data/edges.parquet.")
    try:
        data = {key: (pd.read_parquet(root / path) if key == "edges" else
                      pd.read_csv(root / path, dtype={"gid": "string"})) for key, path in FILES.items()}
        for key, frame in data.items():
            missing = set(REQUIRED[key]) - set(frame.columns)
            if missing:
                raise ValueError(f"{FILES[key]}: missing columns: {', '.join(sorted(missing))}.")
            if frame[list(REQUIRED[key])].isna().any().any():
                raise ValueError(f"{FILES[key]}: required values are missing.")
            numbers = frame.select_dtypes(include="number").to_numpy()
            if np.iscomplexobj(numbers) or not np.isfinite(numbers).all():
                raise ValueError(f"{FILES[key]}: non-finite numeric values.")
        n, c, t, e = (data[key] for key in ("nodes", "clusters", "top", "edges"))
        for frame, columns in ((n, ["gid"]), (t, ["gid"]), (e, ["src", "dst"])):
            for column in columns:
                frame[column] = frame[column].map(lambda v: str(integer_input(v, column))).astype("string")
        for frame, columns in ((n, ["cluster_id", "depth"]), (c, ["cluster_id", "n_nodes", "n_seed"]),
                               (t, ["rank"]), (e, ["n_tx"])):
            for column in columns:
                if not pd.api.types.is_integer_dtype(frame[column]) or (frame[column] < 0).any():
                    raise ValueError(f"Invalid integer column: {column}.")
        for column in ("is_seed", "truncated_by_depth"):
            if not pd.api.types.is_bool_dtype(n[column]):
                raise ValueError(f"Invalid boolean column: {column}.")
        for frame, columns in ((n, ["role_score", "priority_score"]), (t, ["priority_score"])):
            for column in columns:
                if not pd.api.types.is_numeric_dtype(frame[column]) or not frame[column].between(0, 1).all():
                    raise ValueError(f"Invalid score column: {column}.")
        for frame, column in ((n, "evidence"), (c, "hypothesis"), (t, "why")):
            if not frame[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"Empty or invalid {column}.")
        if n.empty or c.empty or t.empty or not n.gid.is_unique or not c.cluster_id.is_unique:
            raise ValueError("Empty or duplicate node/cluster records.")
        if not n.role.isin(ROLES).all() or not t.role.isin(ROLES).all():
            raise ValueError("Unknown role in analytics outputs.")
        if set(n.cluster_id) != set(c.cluster_id) or not (set(e.src) | set(e.dst)) <= set(n.gid):
            raise ValueError("Unknown node or cluster references.")
        if e.duplicated(["src", "dst"]).any() or (e.n_tx <= 0).any() or (e.sum_kzt < 0).any():
            raise ValueError("Invalid observed edges.")
        for row in c.itertuples(index=False):
            members = n[n.cluster_id == row.cluster_id]
            leaders = json.loads(row.top_gids)
            if not isinstance(leaders, list) or not leaders or not all(
                    str(integer_input(gid, "top_gid")) in set(members.gid) for gid in leaders):
                raise ValueError("Invalid cluster top_gids.")
            if row.n_nodes != len(members) or row.n_seed != int(members.is_seed.sum()):
                raise ValueError("Cluster counts do not match members.")
        if not t.gid.is_unique or not set(t.gid) <= set(n.gid) or t['rank'].tolist() != list(range(1, len(t) + 1)):
            raise ValueError("Invalid top priority references or ranks.")
        expected = n.assign(_gid=n.gid.map(int)).sort_values(["priority_score", "_gid"], ascending=[False, True]).head(len(t))
        if t.gid.tolist() != expected.gid.tolist() or t.role.tolist() != expected.role.tolist() or not np.allclose(
                t.priority_score, expected.priority_score, rtol=0, atol=1e-10):
            raise ValueError("Top priority does not match node outputs.")
        return data
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as exc:
        raise ValueError(f"Cannot load presentation data: {exc}") from None


def summary(data):
    n = data["nodes"]
    return {"Nodes": len(n), "Edges": len(data["edges"]),
            "Transactions": int(data["edges"].n_tx.sum()), "Clusters": len(data["clusters"]),
            "Seed nodes": int(n.is_seed.sum()), "Truncated depth-4 nodes": int(n.truncated_by_depth.sum())}


def role_counts(nodes):
    return {role: int(nodes.role.eq(role).sum()) for role in ROLES}


class Analyst:
    """Five retrieval tools and an explicit, deterministic dispatcher."""

    def __init__(self, data):
        self.data = data

    def inspect_node(self, gid):
        gid = str(integer_input(gid, "gid"))
        rows = self.data["nodes"][self.data["nodes"].gid == gid]
        if rows.empty:
            raise ValueError(f"Unknown gid: {gid}.")
        return rows.to_dict("records")[0]

    def inspect_cluster(self, cluster_id):
        cid = integer_input(cluster_id, "cluster_id")
        rows = self.data["clusters"][self.data["clusters"].cluster_id == cid]
        if rows.empty:
            raise ValueError(f"Unknown cluster_id: {cid}.")
        members = self.data["nodes"][self.data["nodes"].cluster_id == cid]
        members = members.assign(_gid=members.gid.map(int)).sort_values(
            ["priority_score", "_gid"], ascending=[False, True]).drop(columns="_gid")
        return {"cluster": rows.to_dict("records")[0], "members": members.to_dict("records"),
                "role_distribution": role_counts(members)}

    def get_top_priority(self, n):
        n = integer_input(n, "n")
        if not 1 <= n <= len(self.data["top"]):
            raise ValueError(f"n must be between 1 and {len(self.data['top'])} (available shortlist).")
        return self.data["top"].head(n).to_dict("records")

    def get_neighbors(self, gid):
        node = self.inspect_node(gid)
        edges = self.data["edges"]
        result = []
        for direction, endpoint, neighbor in (("incoming", "dst", "src"), ("outgoing", "src", "dst")):
            selected = edges[edges[endpoint] == node["gid"]]
            selected = selected.assign(_gid=selected[neighbor].map(int)).sort_values("_gid")
            for row in selected.to_dict("records"):
                result.append({"direction": direction, "neighbor_gid": row[neighbor],
                               "sum_kzt": row["sum_kzt"], "n_tx": row["n_tx"]})
        return result

    def explain_node(self, gid):
        node = self.inspect_node(gid)
        limitations = []
        if node["is_seed"]:
            limitations.append(SEED_WARNING)
        if node["truncated_by_depth"]:
            limitations.append(TRUNCATION_WARNING)
        explanation = (f"gid {node['gid']} is observed as {node['role']} "
                       f"(role_score={node['role_score']:.4f}). Evidence: {node['evidence']}. "
                       f"Review priority={node['priority_score']:.4f}. "
                       "Priority is a review priority, not a probability or proof of wrongdoing.")
        return {"gid": node["gid"], "role": node["role"], "role_score": node["role_score"],
                "explanation": explanation,
                "priority_score": node["priority_score"], "evidence": node["evidence"],
                "diagnostics": {key: node[key] for key in DIAGNOSTICS if key in node},
                "limitations": limitations, "interpretation": PRIORITY_NOTE}

    def dispatch(self, action, parameter):
        trace = [f"OBSERVE: action={action}; parameter={parameter}"]
        actions = {"inspect_node": self.inspect_node, "inspect_cluster": self.inspect_cluster,
                   "get_top_priority": self.get_top_priority, "get_neighbors": self.get_neighbors,
                   "explain_node": self.explain_node}
        if action not in actions:
            return {"ok": False, "error": "Unknown action.", "trace": trace + ["VALIDATE: action rejected"]}
        trace.extend([f"INTERPRET: explicit {action} request", f"DISPATCH: {action}"])
        try:
            result = actions[action](parameter)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "trace": trace + [f"VALIDATE: rejected; {exc}"]}
        trace.extend(["VALIDATE: exact integer and requested record/range verified",
                      "RETRIEVE: verified loaded data returned", f"RESULT: {action} completed"])
        return {"ok": True, "data": result, "trace": trace}
