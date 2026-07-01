"""
Eval / PART 2 — ranking metrics over the hand-labeled gold set.

Relevance = `label_tier` (0-4), graded for NDCG; binary (tier>=3 = relevant) for MAP and P@10;
honeypots (tier 0) are irrelevant. The gold set is the universe: a candidate score column induces a
ranking of the gold rows, which is scored against the labels.

composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10   (the challenge's official weights)

READY TO RUN once labels exist — do NOT run before then (no labels = meaningless).
Optional cross-check against sklearn.metrics.ndcg_score is included.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RELEVANT_TIER = 3  # tier >= 3 counts as "relevant" (per submission_spec: P@10 relevance = tier 3+)


def dcg_at_k(gains, k):
    gains = np.asarray(gains, dtype=float)[:k]
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    return float(np.sum((2 ** gains - 1) * discounts))


def ndcg_at_k(rels_in_rank_order, k):
    """Graded NDCG@k. rels_in_rank_order: relevance grades (label_tier) in ranked order."""
    dcg = dcg_at_k(rels_in_rank_order, k)
    ideal = sorted(rels_in_rank_order, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return (dcg / idcg) if idcg > 0 else 0.0


def average_precision(binary_in_rank_order):
    """AP for one ranking: mean of precision@i at each relevant position."""
    b = np.asarray(binary_in_rank_order, dtype=int)
    n_rel = int(b.sum())
    if n_rel == 0:
        return 0.0
    hits = 0
    precs = []
    for i, r in enumerate(b, start=1):
        if r:
            hits += 1
            precs.append(hits / i)
    return float(np.mean(precs))


def precision_at_k(binary_in_rank_order, k):
    b = np.asarray(binary_in_rank_order, dtype=int)[:k]
    return float(b.sum() / k) if k > 0 else 0.0


def rank_labels(gold_df, score_col, label_col="label_tier"):
    """Order the gold rows by score desc (tie-break candidate_id asc), return the label_tier list."""
    d = gold_df.copy()
    if d[label_col].isna().any():
        raise ValueError("label_tier has blanks — supply independent labels before evaluating.")
    d = d.sort_values([score_col, "candidate_id"], ascending=[False, True])
    return d[label_col].astype(float).tolist()


def evaluate(gold_df, score_col, label_col="label_tier"):
    """Return the metric dict for one score column over the gold set."""
    rels = rank_labels(gold_df, score_col, label_col)            # graded, in ranked order
    binary = [1 if r >= RELEVANT_TIER else 0 for r in rels]
    m = {
        "n": len(rels),
        "n_relevant(tier>=3)": int(sum(binary)),
        "NDCG@10": ndcg_at_k(rels, 10),
        "NDCG@50": ndcg_at_k(rels, 50),
        "MAP": average_precision(binary),
        "P@10": precision_at_k(binary, 10),
    }
    m["composite"] = (0.50 * m["NDCG@10"] + 0.30 * m["NDCG@50"]
                      + 0.15 * m["MAP"] + 0.05 * m["P@10"])
    return m


def evaluate_many(gold_df, score_cols, label_col="label_tier"):
    """Evaluate several score columns; return a tidy DataFrame (one row per column)."""
    rows = []
    for c in score_cols:
        sub = gold_df.dropna(subset=[c])
        rows.append({"score": c, **evaluate(sub, c, label_col)})
    return pd.DataFrame(rows).set_index("score")


def _sklearn_ndcg_check(gold_df, score_col, k, label_col="label_tier"):
    """Optional cross-check using sklearn.metrics.ndcg_score."""
    from sklearn.metrics import ndcg_score
    d = gold_df.dropna(subset=[label_col]).copy()
    y_true = d[label_col].to_numpy().reshape(1, -1)
    y_score = d[score_col].to_numpy().reshape(1, -1)
    return float(ndcg_score(y_true, y_score, k=k))


if __name__ == "__main__":
    print("metrics.py is a library. Import evaluate/evaluate_many after labels are filled in.\n"
          "Example:\n"
          "  import pandas as pd, json\n"
          "  gold = pd.DataFrame([json.loads(l) for l in open('eval/gold_set.jsonl')])\n"
          "  scores = pd.read_csv('eval/ablation_scores.csv')\n"
          "  g = gold.merge(scores, on='candidate_id')\n"
          "  from eval.metrics import evaluate_many\n"
          "  print(evaluate_many(g, ['full','judge_only','embed_only','pos_only','no_availability']))")
