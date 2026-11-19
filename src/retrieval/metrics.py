"""Retrieval metrics: MRR and Recall@5 (capped), overall and per query type."""
import numpy as np
import pandas as pd


def _rel_set(s):
    return set(int(x) for x in str(s).split("|"))


def evaluate(rank_fn, queries, k=5):
    """rank_fn(query_text) -> list of recipe_ids ranked best-first (full ranking).

    MRR       : mean of 1 / rank of the first relevant recipe (over the full ranking).
    Recall@k  : |relevant in top-k| / min(k, |relevant|)  (capped, so 1.0 is reachable
                even when a query has more than k relevant recipes).
    """
    rows = []
    for _, q in queries.iterrows():
        rel = _rel_set(q["relevant_ids"])
        ranked = [int(i) for i in rank_fn(q["query"])]
        rr = next((1.0 / (r + 1) for r, i in enumerate(ranked) if i in rel), 0.0)
        hits = len(rel & set(ranked[:k]))
        rows.append({"query_id": q["query_id"], "type": q["type"], "query": q["query"],
                     "rr": rr, f"recall@{k}": hits / min(k, len(rel))})
    per_q = pd.DataFrame(rows)
    summary = {"MRR": float(per_q["rr"].mean()), f"Recall@{k}": float(per_q[f"recall@{k}"].mean()),
               "n_queries": int(len(per_q))}
    by_type = per_q.groupby("type")[["rr", f"recall@{k}"]].mean().rename(columns={"rr": "MRR"})
    return summary, by_type, per_q
